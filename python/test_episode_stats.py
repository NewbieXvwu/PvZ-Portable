"""Identity statistics stay valid across reordered decks and moving conveyor slots."""
import unittest
from pvz_episode_stats import planted_by_type, planted_slots, plant_type_before_action


class EpisodeStatsTest(unittest.TestCase):
    def test_new_types_win_over_legacy_and_conveyor(self):
        record={'planted_types':{'9':74,'11':7},'planted_slots':{'8':74,'9':7},'planted':{'9':1000}}
        self.assertEqual(planted_by_type(record,[],level=20,conveyor_start='original'),{'9':74,'11':7})
        self.assertEqual(planted_slots(record),{'8':74,'9':7})

    def test_old_n9_maps_slot8_to_sunshroom(self):
        self.assertEqual(planted_by_type({'planted':{'8':74}},[0,1,2,3,4,5,7,8,9],level=14),{'9':74})

    def test_old_n10_maps_slot9_to_gravebuster(self):
        self.assertEqual(planted_by_type({'planted':{'9':7}},[0,1,2,3,4,5,7,8,9,11],level=14),{'11':7})

    def test_per_game_reordered_deck_and_duplicates(self):
        game={'planted':{'0':2,'1':3,'2':4}}
        self.assertEqual(planted_by_type(game,[9,1,9],level=7),{'1':3,'9':6})

    def test_old_conveyor_is_unknown_even_with_a_full_initial_deck(self):
        for level in (10,20,25,30):
            with self.subTest(level=level):
                self.assertIsNone(planted_by_type({'planted':{'0':3}},[0,1,2,3,4,5,7],level=level))
        self.assertIsNone(planted_by_type({'planted':{}},[],conveyor_start='original'))

    def test_task_dictionary_context(self):
        self.assertIsNone(planted_by_type({'planted':{'0':1}},{'deck':[1,0,2,3,4,5],'level':10}))
        self.assertEqual(planted_by_type({'planted':{'0':1}},{'deck':[1,0,2,3,4,5],'level':7}),{'1':1})

    def test_intermediate_task27_identity_counts_are_supported(self):
        self.assertEqual(planted_by_type({'planted_by_type':{'11':3},'planted':{'9':3}},[],level=20),{'11':3})
        self.assertIsNone(planted_by_type({'planted_by_type':[0]*10},[0,1,2,3,4,5,7],level=7))

    def test_unmappable_or_missing_old_data_is_unknown(self):
        self.assertIsNone(planted_by_type({'planted':{'9':1}},[0,1,2,3,4,5,7],level=7))
        self.assertIsNone(planted_by_type({},[0,1,2,3,4,5,7],level=7))
        self.assertEqual(planted_by_type({'planted':{}},[0,1,2,3,4,5,7],level=7),{})

    def test_native_slot_array_is_explicitly_supported(self):
        self.assertEqual(planted_slots({'planted_slots':[0,2,0,4]}),{'1':2,'3':4})

    def test_bad_counts_fail_instead_of_guessing(self):
        for value in ({'9':-1},{'9':True},{'bad':2}):
            with self.subTest(value=value),self.assertRaises(ValueError):
                planted_by_type({'planted_types':value},[])
        with self.assertRaises(ValueError):planted_by_type({'planted_types':[0,1]},[])

    def test_live_type_is_captured_before_slot_moves(self):
        before={'packets':[{'index':0,'type':11},{'index':1,'type':9}]}
        after={'packets':[{'index':0,'type':9}]}
        self.assertEqual(plant_type_before_action(before,0),11)
        self.assertEqual(plant_type_before_action(after,0),9)
        self.assertEqual(plant_type_before_action({'packets':[{'index':0,'type':48,'imitater_type':3}]},0),3)


if __name__=='__main__':unittest.main()
