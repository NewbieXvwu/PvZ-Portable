"""Public edge selection and fail-closed release contracts; no native/game claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_event_wait_dynamic_audit import public_edges, check_protocol
from test_wait_events import observation


class DynamicNativeContracts(unittest.TestCase):
    def test_transient_edges_use_public_packet_wave_and_ready_defense_fields(self):
        before=observation(720)
        before['packets'][0]['active']=False
        after=copy.deepcopy(before)
        after['tick']+=60;after['wave']+=1
        after['packets'][0]['active']=True
        after['defenses'][0]['state']=2
        self.assertEqual(public_edges(before,after),[
            ('defense_lost','timeout'),('condition:wave_changed','wave_changed'),
            ('condition:packet_became_ready','packet_became_ready')])

    def test_future_roster_or_cooldown_change_without_readiness_is_not_an_edge(self):
        before=observation(0)
        after=copy.deepcopy(before)
        after['tick']=60
        after['future_wave_zombies']=[999,998]
        after['packets'][0].update(cooldown=750,refresh_time=750)
        self.assertEqual(public_edges(before,after),[])
        before['defenses'][0]['state']=2
        after['defenses'][0]['state']=1
        self.assertEqual(public_edges(before,after),[])

    def test_draft_precedes_any_publication_source_binary_or_original_run_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'draft.json';path.write_text(json.dumps({'release_status':'draft'}))
            with self.assertRaisesRegex(RuntimeError,'released published protocol'):
                check_protocol(path)

    def test_invalid_public_defense_state_is_not_silently_coerced(self):
        before=observation(0);after=copy.deepcopy(before)
        after['defenses'][0]['state']=True
        with self.assertRaisesRegex(ValueError,'public integer'):
            public_edges(before,after)


if __name__=='__main__':unittest.main()
