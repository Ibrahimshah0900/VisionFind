from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import video_chat

class VideoChatTests(unittest.TestCase):
    def test_dispatch_preserves_image_followups(self):
        self.assertFalse(video_chat.handles({'text':'Describe it'}, {'images':[{}], 'video':{}}))
        self.assertFalse(video_chat.handles({'files':['image.jpg']}, {'video':{'path':'v.mp4'}}))
        self.assertTrue(video_chat.handles({'files':['video.mp4']}, {}))
        self.assertTrue(video_chat.handles({'text':'Explain this video'}, {'video':{'path':'v.mp4'}, 'images':[{}]}))
        self.assertTrue(video_chat.handles({'text':'Draw boxes around people'}, {'video':{'path':'v.mp4'}}))

    def test_cached_box_response_has_no_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            boxed, data = root/'boxes.mp4', root/'detections.json'
            boxed.write_bytes(b'placeholder')
            data.write_text(json.dumps([{'frame_index':0,'timestamp_seconds':0,'detected_count':3,'detections':[{'label':'person'}]*3}]))
            cache = root/'results/video_chat_cache'
            cache.mkdir(parents=True)
            (cache/'abc.json').write_text(json.dumps({'boxed_video':str(boxed), 'detections':str(data)}))
            state = {'video':{'path':'existing.mp4', 'sha256':'abc'}, 'selected_ids':[2]}
            with patch.object(video_chat.gr, 'Video', return_value=SimpleNamespace(value=str(boxed))), \
                 patch.object(video_chat, 'render_people', side_effect=AssertionError('Detector rerun')):
                events = list(video_chat.respond(SimpleNamespace(root=root),
                    {'text':'Count people and draw boxes around them'}, [], state))
            messages, final_state = events[-1][1:]
            self.assertEqual(final_state['selected_ids'], [2])
            self.assertTrue(any('3 to 3' in m['content'] for m in messages if isinstance(m['content'], str)))
            self.assertTrue(any(getattr(m['content'], 'value', None)==str(boxed) for m in messages))
            self.assertEqual(state['selected_ids'], [2])

    def test_unsupported_target_clarifies_without_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            events = list(video_chat.respond(SimpleNamespace(root=folder),
                {'text':'Draw boxes around spaceships'}, [], {'video':{'path':'v.mp4','sha256':'x'}}))
            self.assertIn('Supported video targets', events[-1][1][-1]['content'])

    def test_combined_request_returns_description_and_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            boxed = root/'boxes.mp4'
            boxed.write_bytes(b'placeholder')
            report = {'observations':[{'timestamp_seconds':0.0, 'answer':{'answer':'People are visible.'}}]}
            with patch.object(video_chat.video, 'analyze_video', return_value=report) as describe, \
                 patch.object(video_chat, 'render_people', return_value=(boxed,[{'detected_count':2}])), \
                 patch.object(video_chat.gr, 'Video', return_value=SimpleNamespace(value=str(boxed))):
                events = list(video_chat.respond(SimpleNamespace(root=root),
                    {'text':'Explain this video and draw boxes around people'}, [],
                    {'video':{'path':'v.mp4','sha256':'abc'}}))
            messages = events[-1][1]
            describe.assert_called_once()
            self.assertTrue(any('People are visible.' in m['content'] for m in messages if isinstance(m['content'],str)))
            self.assertTrue(any(getattr(m['content'],'value',None)==str(boxed) for m in messages))

if __name__ == '__main__':
    unittest.main()
