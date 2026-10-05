import json,sys,torch
from pathlib import Path
sys.path.insert(0,'/tmp')
sys.path[:0]=['/Users/newbiexvwu/PvZAgent/python','/Users/newbiexvwu/PvZAgent/scripts']
import pvz_bc_train_eval as trainer
from pvz_agent_model import GameplayModelV1
out=Path('/tmp/pvz_bc_2b')
checkpoint=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False)
model=GameplayModelV1(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
torch.set_num_threads(4)
teacher=trainer.importlib_teacher()
seen=trainer.run_eval(model,teacher,range(trainer.TRAIN_FIRST,trainer.TRAIN_FIRST+trainer.SEEN_COUNT),True)
unseen=trainer.run_eval(model,teacher,range(trainer.UNSEEN_FIRST,trainer.UNSEEN_FIRST+trainer.UNSEEN_COUNT),True)
greedy=trainer.run_eval(model,teacher,range(trainer.UNSEEN_FIRST,trainer.UNSEEN_FIRST+trainer.UNSEEN_COUNT),False)
report={'evaluation_device':'cpu','sampled_seen':seen,'sampled_unseen':unseen,'greedy_unseen':greedy,
        'greedy_caveat':'structurally invalid per handoff: factored action argmax favors wait'}
(out/'evaluation_results.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({'phase':'evaluation_done','results':report},ensure_ascii=False),flush=True)
