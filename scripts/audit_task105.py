#!/usr/bin/env python3
"""Compatibility entry for the already-running chain; only assemble existing results.

The extra audit/reimport/replay chain was removed by the controller's task106 rule.
"""
from task105_results import main

if __name__ == '__main__':
    main()
