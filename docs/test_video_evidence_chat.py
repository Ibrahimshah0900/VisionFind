from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import video_requests, video_chat

class EvidenceChatTests(unittest.TestCase):
    def test_timing_and_count_routing(self):
        for question in ['When was a car detected?','How many cars were detected?','Show a car detection timeline']:
            plan = video_requests.plan_video_request(question)
            self.assertTrue(plan['evidence_summary'])
            self.assertFalse(plan['clarification'])
            self.assertEqual(plan['targets'],['car'])
        self.assertTrue(video_requests.plan_video_request('When was it detected?')['clarification'])

    def test_cached_timeline_needs_no_box_file_or_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'detections.json'
            data.write_text(json.dumps([
                {'frame_index':0,'timestamp_seconds':0,'detected_count':0,'detections':[]},
                {'frame_index':1,'timestamp_seconds':.1,'detected_count':1,'detections':[{'label':'car'}]}]))
            plan = video_requests.plan_video_request('When was a car detected?')
            cache = root/'results/video_chat_cache'
            cache.mkdir(parents=True)
            key = video_chat.cache_key('abc',plan)
            (cache/(key+'.json')).write_text(json.dumps({'detections':str(data),'boxed_video':str(root/'missing.mp4')}))
            with patch.object(video_chat,'render_people',side_effect=AssertionError('Inference should not run')):
                events = list(video_chat.respond(SimpleNamespace(root=root),
                    {'text':'When was a car detected?'},[],{'video':{'path':'v.mp4','sha256':'abc'}}))
            texts = [m['content'] for m in events[-1][1] if isinstance(m['content'],str)]
            self.assertTrue(any('0.10' in text and 'prediction runs' in text for text in texts))
            self.assertFalse(any('could not complete' in text for text in texts))

if __name__ == '__main__':
    unittest.main()
