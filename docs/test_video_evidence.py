from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.video_evidence import summarize, render

def frame(index,time,labels):
    return {'frame_index':index,'timestamp_seconds':time,
            'detected_count':len(labels),'detections':[{'label':label} for label in labels]}

class EvidenceTests(unittest.TestCase):
    def test_does_not_bridge_unobserved_or_empty_frames(self):
        result = summarize([frame(0,0,['car']),frame(1,.1,['car']),
            frame(2,.2,[]),frame(3,.3,['car']),frame(5,.5,['car'])],['car'])
        self.assertEqual([r['frames'] for r in result['categories'][0]['prediction_runs']],[2,1,1])
        self.assertEqual(result['categories'][0]['frames_with_predictions'],4)

    def test_empty_category_is_not_claimed_absent(self):
        result = summarize([frame(0,0,[])],['bottle'])
        self.assertIn('does not establish absence',render(result))
        self.assertEqual(result['categories'][0]['prediction_runs'],[])

    def test_invalid_evidence_rejected(self):
        for records in ([frame(0,0,[]),frame(1,0,[])],
                        [frame(0,float('nan'),[])],
                        [{'frame_index':0,'timestamp_seconds':0,'detected_count':2,'detections':[]} ]):
            with self.assertRaises(ValueError):
                summarize(records,['car'])

if __name__ == '__main__':
    unittest.main()
