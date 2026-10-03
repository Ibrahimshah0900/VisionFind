from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import video_chat, video_object_detector, detector_candidate

class StrongerVideoIntegrationTests(unittest.TestCase):
    def test_adapter_forwards_categories(self):
        result = {'detected_count':2,'detections':[]}
        with patch.object(detector_candidate,'detect',return_value=result) as detect:
            self.assertEqual(video_object_detector.detect_video_objects('frame.jpg',['car','bicycle']),result)
            detect.assert_called_once_with('frame.jpg',['car','bicycle'])

    def test_release_delegates(self):
        with patch.object(detector_candidate,'release') as release:
            video_object_detector.release()
            release.assert_called_once()

    def test_caches_separate_backend_and_targets(self):
        person = video_chat.cache_key('hash',{'targets':['person']})
        car = video_chat.cache_key('hash',{'targets':['car']})
        mixed = video_chat.cache_key('hash',{'targets':['bicycle','car']})
        self.assertEqual(person,'hash')
        self.assertNotEqual(car,'hash_car')
        self.assertIn(video_object_detector.BACKEND_VERSION,car)
        self.assertNotEqual(car,mixed)
        self.assertEqual(mixed,video_chat.cache_key('hash',{'targets':['car','bicycle']}))

if __name__ == '__main__':
    unittest.main()
