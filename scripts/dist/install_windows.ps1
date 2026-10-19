# PvZ distributed collection - Windows installer (WSL2 + Ubuntu + sshd + autostart).
#
# Right-click -> "Run with PowerShell" as Administrator, or:
#     powershell -ExecutionPolicy Bypass -File install_windows.ps1
#
# It is idempotent and survives the first reboot: it writes a stage marker to
# C:\ProgramData\pvz-worker\stage.txt and, when Windows needs a restart, registers a
# RunOnce entry that continues automatically after the next logon.
#
# What it does
#   1. enable WSL2 + VirtualMachinePlatform, install Ubuntu (needs a restart the first time)
#   2. inside Ubuntu: openssh-server on port 2222, our public key for root, sshd on boot
#   3. logon task that starts the collection worker, desktop shortcut to its web page
#
# Uninstall:  powershell -ExecutionPolicy Bypass -File install_windows.ps1 -Uninstall
# (this runs "wsl --unregister Ubuntu" - it deletes that Ubuntu distro and everything in it)
#
# All output is ASCII on purpose: Windows PowerShell 5.1 reads BOM-less files as ANSI.
param(
    [string]$Distro = "Ubuntu",
    [int]$Port = 2222,
    [switch]$Uninstall,
    [switch]$ProvisionOnly
)

$ErrorActionPreference = "Stop"
$StateDir = "C:\ProgramData\pvz-worker"
$StageFile = Join-Path $StateDir "stage.txt"
$StartCmd = Join-Path $StateDir "start-worker.cmd"
$TaskName = "PvZ Worker"
$PublicKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINTPisC+P7urgNbfcXuH6EeU6MtW3E4dh4OME3FG8C9/ newbiexvwu@gmail.com"
$NL = [Environment]::NewLine

function Say($msg) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) }

function Require-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Say "Elevating (a UAC prompt will appear)..."
        $arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $PSCommandPath + '"'
        if ($Distro -ne "Ubuntu") { $arguments += " -Distro $Distro" }
        if ($Port -ne 2222) { $arguments += " -Port $Port" }
        if ($Uninstall) { $arguments += " -Uninstall" }
        if ($ProvisionOnly) { $arguments += " -ProvisionOnly" }
        Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
        exit 0
    }
}

function Get-Stage {
    if (Test-Path $StageFile) { return (Get-Content $StageFile -Raw).Trim() }
    return "start"
}

function Set-Stage($value) {
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    Set-Content -Path $StageFile -Value $value -Encoding ASCII
}

function Test-Distro {
    # authoritative check: does WSL actually start this distro as root?
    # (parsing "wsl -l -q" is unreliable: the output is UTF-16 and locale dependent)
    & wsl.exe -d $Distro -u root -e true 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Restart-Later {
    Set-Stage "wsl-ready"
    $cmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $PSCommandPath + '" -Distro ' + $Distro + ' -Port ' + $Port
    New-Item -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce" -Force | Out-Null
    Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce" -Name "PvZWorkerSetup" -Value $cmd
}

function Invoke-Uninstall {
    Say "Removing the logon task and the desktop shortcut"
    & schtasks.exe /Delete /TN $TaskName /F 2>$null | Out-Null
    $shortcut = Join-Path $env:PUBLIC "Desktop\PvZ progress.url"
    if (Test-Path $shortcut) { Remove-Item $shortcut -Force }
    if (Test-Path $StartCmd) { Remove-Item $StartCmd -Force }
    Say "Unregistering the WSL distro '$Distro' (all data in it is deleted)"
    & wsl.exe --terminate $Distro 2>$null | Out-Null
    & wsl.exe --unregister $Distro
    if (Test-Path $StateDir) { Remove-Item $StateDir -Recurse -Force }
    Say "Done. WSL itself is still installed; remove it from 'Apps and features' if you want."
    exit 0
}

Require-Admin
if ($Uninstall) { Invoke-Uninstall }

$Stage = Get-Stage
if ($Stage -eq "after-reboot") {
    # the RunOnce entry brought us back here; features are already enabled
    Set-Stage "wsl-ready"
    $Stage = "wsl-ready"
}
Say "PvZ worker installer: distro=$Distro ssh-port=$Port stage=$Stage"

# ---------------------------------------------------------------- stage 1: WSL + Ubuntu
if ($Stage -eq "start" -and -not $ProvisionOnly) {
    Say "Enabling WSL and VirtualMachinePlatform"
    $reboot = $false
    foreach ($feature in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
        $state = (Get-WindowsOptionalFeature -Online -FeatureName $feature).State
        if ($state -ne "Enabled") {
            Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart | Out-Null
            $reboot = $true
            Say "  enabled $feature"
        } else {
            Say "  $feature already enabled"
        }
    }
    if ($reboot) {
        Say "Windows needs a restart to finish enabling WSL."
        Set-Stage "after-reboot"
        Restart-Later
        Say "The installer will continue automatically after you log back in."
        Say "Please restart the computer now (Start menu -> Restart)."
        Read-Host "Press Enter to restart now, or close this window to restart later"
        Restart-Computer -Force
        exit 0
    }
    Set-Stage "wsl-ready"
}

# ---------------------------------------------------------------- stage 2: install the distro
if (-not $ProvisionOnly -and $Stage -eq "wsl-ready") {
    & wsl.exe --update 2>$null | Out-Null
    & wsl.exe --set-default-version 2 2>$null | Out-Null
    if (-not (Test-Distro)) {
        Say "Installing $Distro (this downloads a few hundred MB)"
        & wsl.exe --install -d $Distro --no-launch 2>$null
        if ($LASTEXITCODE -ne 0) {
            # older WSL builds do not know --no-launch: install normally, then close its window
            Say "  --no-launch unsupported here, falling back to a plain install"
            & wsl.exe --install -d $Distro
            if ($LASTEXITCODE -ne 0) {
                Say "wsl --install failed (exit $LASTEXITCODE)."
                Say "  - If it mentions virtualization, enable VT-x/SVM in the BIOS and run this script again."
                Say "  - If it mentions the Store or a download, connect to the internet and run it again."
                throw "wsl --install -d $Distro failed"
            }
            Start-Sleep -Seconds 20
            & wsl.exe --terminate $Distro 2>$null | Out-Null
        }
        $ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            if (Test-Distro) { $ready = $true; break }
            Start-Sleep -Seconds 10
        }
        if (-not $ready) {
            # WSL2 kernel / virtualisation needs a restart before the distro will start
            Say "The distro is installed but WSL cannot start it yet; a restart is needed."
            Restart-Later
            Say "The installer will continue automatically after you log back in."
            Read-Host "Press Enter to restart now, or close this window to restart later"
            Restart-Computer -Force
            exit 0
        }
    } else {
        Say "$Distro already installed and starts correctly"
    }
    Set-Stage "provision"
}

# ---------------------------------------------------------------- stage 3: provision inside WSL
if ($Stage -eq "provision" -or $ProvisionOnly) {
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    $bashPath = Join-Path $StateDir "provision.sh"
    $lines = @(
        "set -e",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -qq",
        "apt-get install -y -qq openssh-server curl git ca-certificates >/dev/null",
        "mkdir -p /root/.ssh",
        "chmod 700 /root/.ssh",
        "touch /root/.ssh/authorized_keys",
        'grep -q "AAAAC3NzaC1lZDI1NTE5AAAAINTPisC+P7urgNbfcXuH6EeU6MtW3E4dh4OME3FG8C9/" /root/.ssh/authorized_keys || echo "' + $PublicKey + '" >> /root/.ssh/authorized_keys',
        "chmod 600 /root/.ssh/authorized_keys",
        "sed -i 's/^#*Port .*/Port $Port/' /etc/ssh/sshd_config",
        "sed -i 's/^#*PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config",
        "sed -i 's/^#*PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config",
        # sshd on distro boot: keep an existing [boot] section working, do not add a second one
        "if ! grep -q '^\[boot\]' /etc/wsl.conf 2>/dev/null; then printf '\n[boot]\ncommand = /usr/sbin/service ssh start\n' >> /etc/wsl.conf; elif ! grep -q '^command *=' /etc/wsl.conf; then sed -i '/^\[boot\]/a command = /usr/sbin/service ssh start' /etc/wsl.conf; fi",
        "service ssh restart",
        "sleep 1",
        # fail loudly here instead of letting the controller discover it over a slow tunnel
        "(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':$Port ' || { echo 'sshd is not listening on port $Port'; exit 1; }",
        "service ssh status | head -2"
    )
    [IO.File]::WriteAllText($bashPath, ($lines -join $NL), (New-Object System.Text.UTF8Encoding($false)))
    $unixPath = "/mnt/" + $bashPath.Substring(0, 1).ToLower() + $bashPath.Substring(2).Replace("\", "/")
    Say "Provisioning sshd inside $Distro"
    & wsl.exe --terminate $Distro 2>$null | Out-Null
    & wsl.exe -d $Distro -u root -e bash $unixPath
    if ($LASTEXITCODE -ne 0) {
        Say "Provisioning failed (exit $LASTEXITCODE)."
        Say "  Check that WSL works:  wsl -d $Distro -u root -e bash -c 'echo ok'"
        Say "  Then run this script again (it is safe to repeat)."
        throw "provisioning failed"
    }
    Set-Stage "autostart"
}

# ---------------------------------------------------------------- stage 4: autostart + shortcut
if ($Stage -eq "autostart" -or $ProvisionOnly) {
    $cmdBody = @(
        "@echo off",
        'wsl.exe -d ' + $Distro + ' -u root -e bash -lc "cd /root/pvz-worker && mkdir -p logs && setsid nohup ./env/bin/python repo/scripts/dist/worker.py --root /root/pvz-worker run >> logs/worker.out 2>&1 < /dev/null & sleep 1; true"'
    )
    [IO.File]::WriteAllText($StartCmd, ($cmdBody -join $NL), (New-Object System.Text.UTF8Encoding($false)))
    Say "Registering the logon task '$TaskName'"
    & schtasks.exe /Create /SC ONLOGON /TN $TaskName /TR ('"' + $StartCmd + '"') /RL HIGHEST /F | Out-Null
    $shortcut = Join-Path $env:PUBLIC "Desktop\PvZ progress.url"
    $url = "[InternetShortcut]" + $NL + "URL=http://localhost:8765/" + $NL
    [IO.File]::WriteAllText($shortcut, $url, (New-Object System.Text.UTF8Encoding($false)))
    Set-Stage "done"
}

Say ""
Say "Install finished."
Say "  ssh from the tunnel:  ssh -p $Port root@localhost   (on this PC; the tunnel client forwards 127.0.0.1:$Port)"
Say "  progress page:        http://localhost:8765/        (also the desktop shortcut)"
Say "  worker logs:          \\wsl$\$Distro\root\pvz-worker\logs\"
Say "  the worker waits until the controller pushes code and models; until then it does nothing."
Say "  to remove everything: powershell -ExecutionPolicy Bypass -File install_windows.ps1 -Uninstall"
