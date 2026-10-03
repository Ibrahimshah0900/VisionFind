"""Synthetic-data checks for evaluation plumbing, not model performance."""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

path=Path(__file__).with_name('evaluate_fresh_images.py')
spec=importlib.util.spec_from_file_location('fresh_eval_checks',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class EvaluationChecks(unittest.TestCase):
    def test_parse_eighty_names(self):
        text='names:\n'+''.join(f'  {i}: object{i}\n' for i in range(80))
        self.assertEqual(len(module.parse_names(text)),80)
        with self.assertRaises(ValueError): module.parse_names('names:\n  0: person\n')
    def test_selection_frozen_diverse_and_disjoint(self):
        records=[{'coco_image_id':str(i),'primary':f'class{i%20}','classes':[f'class{i%20}']} for i in range(35)]
        selected=module.select_records(records)
        self.assertEqual(selected,module.select_records(records))
        self.assertEqual(len({r['primary'] for r in selected}),20)
        a={r['coco_image_id'] for r in selected[::2]}
        b={r['coco_image_id'] for r in selected[1::2]}
        self.assertEqual(len(a),10)
        self.assertFalse(a&b)
    def test_selection_rejects_insufficient_data(self):
        with self.assertRaises(ValueError): module.select_records([])
    def test_multiple_annotated_matches_are_accepted(self):
        records=[{'primary':'dog','classes':['dog','person']},
                 {'primary':'person','classes':['person','dog']},
                 {'primary':'cat','classes':['cat']}]
        queries={r['category']:r for r in module.category_queries(records)}
        self.assertEqual(queries['dog']['expected_ids'],[1,2])
        self.assertEqual(queries['cat']['expected_ids'],[3])
    def test_annotation_loading_and_largest_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'images/train2017').mkdir(parents=True)
            (root/'labels/train2017').mkdir(parents=True)
            Image.new('RGB',(10,10)).save(root/'images/train2017/123.jpg')
            (root/'labels/train2017/123.txt').write_text('0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.8 0.8\n')
            records=module.read_records(root,{0:'person',1:'dog'})
            self.assertEqual(records[0]['primary'],'dog')
            self.assertEqual(records[0]['classes'],['dog','person'])
            (root/'labels/train2017/123.txt').write_text('1 0.5 0.5 -0.2 0.2\n')
            with self.assertRaises(ValueError): module.read_records(root,{1:'dog'})
    def test_archive_path_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as saved: saved.writestr('../escape.txt','invalid')
            with self.assertRaises(ValueError): module.extract_dataset(archive,root/'output')
            self.assertFalse((root/'escape.txt').exists())

if __name__=='__main__': unittest.main()
