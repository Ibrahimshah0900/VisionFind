"""CPU-only regression checks; no model downloads or GPU required."""
import ast
import re
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'src/multi_chat.py'
tree = ast.parse(source.read_text())
names = {'ORDINALS', 'IMAGE_WORD', 'NUMBER_REFERENCE'}
nodes = [n for n in tree.body if
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
         or isinstance(n, ast.FunctionDef) and n.name in {'select_images', 'analyze_one', 'split_requests'}]
ns = {'re': re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)

class ScopeTests(unittest.TestCase):
    def test_reported_description(self):
        ids, comparison, query = ns['select_images']('Describe the three images attached', [1,2,3], [], [])
        self.assertEqual(ids, [1,2,3])
        self.assertFalse(comparison)
        self.assertEqual(query, 'Describe the image')
    def test_reported_boxes(self):
        ids, _, query = ns['select_images']('draw box around the object found in 3 of the images', [1,2,3], [], [])
        self.assertEqual(ids, [1,2,3])
        self.assertNotIn('3 of', query)
    def test_ambiguous_quantity(self):
        with self.assertRaises(ValueError):
            ns['select_images']('describe three images', [1,2,3,4], [], [])
    def test_explicit_and_followup(self):
        self.assertEqual(ns['select_images']('describe image 3', [1,2,3], [], [1,2])[0], [3])
        self.assertEqual(ns['select_images']('what color is it?', [1,2,3], [2], [])[0], [2])
    def test_multiple_requests(self):
        requests = ns['split_requests']('Describe all images. Compare images 1 and 2. Draw boxes around all people in image 3.')
        self.assertEqual(len(requests), 3)
    def test_count_and_same_target_boxes_merge(self):
        self.assertEqual(len(ns['split_requests']('Count the bowls and draw boxes around all bowls.')), 1)
    def test_count_and_pronoun_boxes_merge(self):
        clauses = ns['split_requests']('Count the bowls and draw bounding boxes around them.')
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0], 'Count the bowls and draw bounding boxes around the bowls')
    def test_pronoun_other_image_stays_separate(self):
        self.assertEqual(len(ns['split_requests']('Count bowls in image 1 and draw boxes around them in image 2.')), 2)
    def test_count_and_other_target_boxes_stay_separate(self):
        self.assertEqual(len(ns['split_requests']('Count bowls and draw boxes around cups.')), 2)
    def test_count_and_other_image_boxes_stay_separate(self):
        self.assertEqual(len(ns['split_requests']('Count bowls in image 1 and draw boxes around bowls in image 2.')), 2)
    def test_merged_count_keeps_next_action(self):
        self.assertEqual(len(ns['split_requests']('Count bowls and draw boxes around bowls. Describe image 2.')), 2)
    def test_visual_prompt_has_no_old_generated_answer(self):
        class Engine:
            def ask(self, path, question, **kwargs):
                self.prompt = question
                return {'answer': 'visible scene'}
        engine = Engine()
        ns['detector_kind'] = lambda *args: None
        ns['analyze_one'](engine, {'id':3, 'path':'test.png'}, 'Describe the image',
            {'mode':'answer', 'locate':False}, None,
            [{'role':'assistant', 'content':'invented rooftop sunset'}])
        self.assertNotIn('rooftop', engine.prompt)
        self.assertIn('exactly one', engine.prompt)
    def test_generic_target_clarifies(self):
        text, media, detail = ns['analyze_one'](None, {'id':1,'path':'test.png'}, 'draw objects',
            {'mode':'answer','locate':True,'target':'objects'}, None, [])
        self.assertTrue(detail['needs_clarification'])
        self.assertEqual(media, [])


class SelectionRegressionTests(unittest.TestCase):
    def test_same_color_is_not_comparison(self):
        self.assertFalse(ns['select_images']('is the dog the same color as the cat?', [1,2], [1], [])[1])
    def test_both_requires_two(self):
        with self.assertRaises(ValueError):
            ns['select_images']('describe both images', [1,2,3], [], [])

class ComposerTests(unittest.TestCase):
    """Uses real Gradio components; model inference is mocked."""
    def setUp(self):
        import copy, hashlib, json, tempfile, types, uuid
        import gradio as gr
        from PIL import Image, ImageOps
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files=[]
        for i, color in enumerate(['red','blue','green']):
            file=self.root/f'{i}.png'
            Image.new('RGB',(64,64),color).save(file)
            self.files.append(str(file))
        self.engine=types.SimpleNamespace(root=self.root)
        self.ns=dict(ns, copy=copy, hashlib=hashlib, json=json, uuid=uuid, Path=Path,
                     gr=gr, Image=Image, ImageOps=ImageOps)
        self.ns['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=MemoryError))
        self.ns['router']=types.SimpleNamespace(normalize_query=lambda x:x,
            route_request=lambda *args: ({'mode':'answer','locate':False},'test'))
        self.ns['analyze_one']=lambda engine,item,*args: ('Test answer',[],{'image_id':item['id']})
        self.ns['compare_images']=lambda *args:'Test comparison'
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
               {'save_json','read_image','build_multi_chat'}]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'), self.ns)
        self.app=self.ns['build_multi_chat'](self.engine)
        self.addCleanup(self.app.close)
        self.respond=self.app.visionfind_chat_handler
    def test_uploaded_search_returns_ids_and_selects_top_candidate(self):
        import sys, types
        from unittest.mock import patch
        self.ns['__package__'] = 'src'
        self.ns['router'].route_request = lambda *args: ({'mode':'search','query':'dog'}, 'mock')
        class Rag:
            cache_status = 'loaded'
            @classmethod
            def from_state(cls, engine, state):
                instance = cls()
                instance.items = state['images']
                return instance
            def search(self, query, top_k):
                return [{'image_id': item['id'], 'path': item['path'], 'similarity': .3}
                        for item in self.items]
        package = types.ModuleType('src'); package.__path__ = []
        module = types.ModuleType('src.upload_rag'); module.UploadedImageRAG = Rag
        with patch.dict(sys.modules, {'src': package, 'src.upload_rag': module}):
            final = self.run_turn('Find photos of a dog', self.files[:2])[-1]
        _, chat, state = final
        texts = [m['content'] for m in chat if isinstance(m['content'], str)]
        self.assertTrue(any('Ranked candidates' in t and 'Image 1' in t and 'Image 2' in t for t in texts))
        self.assertEqual(state['selected_ids'], [1])
    def run_turn(self,text,files=None,chat=None,state=None):
        return list(self.respond({'text':text,'files':files or []},chat or [],state or {}))
    def test_enter_submit_configuration(self):
        import gradio as gr
        composer=next(c for c in self.app.blocks.values() if isinstance(c,gr.MultimodalTextbox))
        self.assertEqual(composer.lines,1)
        self.assertEqual(composer.sources,['upload'])
        self.assertTrue(composer.submit_btn)
        self.assertTrue(any((composer._id,'submit') in f.targets for f in self.app.fns.values()))
    def test_atomic_composer_and_immediate_message(self):
        updates=self.run_turn('Describe the three images attached',self.files)
        self.assertFalse(updates[0][0]['interactive'])
        self.assertTrue(updates[-1][0]['interactive'])
        state=updates[-1][2]
        self.assertEqual(len(state['images']),3)
        import gradio as gr
        galleries=[m['content'] for m in updates[-1][1] if isinstance(m['content'],gr.Gallery)]
        self.assertEqual(len(galleries),1)
        self.assertEqual(galleries[0].height,126)
        chat_component=next(c for c in self.app.blocks.values() if isinstance(c,gr.Chatbot))
        self.assertIsNotNone(chat_component.postprocess(updates[-1][1]))
    def test_duplicate_upload_selects_existing_image(self):
        first=self.run_turn('describe all images',self.files)[-1]
        second=self.run_turn('describe', [self.files[1]],first[1],first[2])[-1]
        self.assertEqual(second[2]['selected_ids'],[2])
        self.assertEqual(len(second[2]['images']),3)
    def test_bad_upload_preserves_payload(self):
        updates=self.run_turn('describe',[str(self.root/'missing.png')])
        self.assertTrue(updates[-1][0]['interactive'])
        self.assertEqual(updates[-1][0]['value']['text'],'describe')
        self.assertIn("couldn't complete",updates[-1][1][-1]['content'])
    def make_ten_files(self):
        from PIL import Image
        files=[]
        for i in range(10):
            path=self.root/f'capacity_{i}.png'
            Image.new('RGB',(48,48),(i*20,40,80)).save(path)
            files.append(str(path))
        return files
    def test_ten_accepted_eleventh_preserves_conversation(self):
        ten=self.run_turn('describe all images',self.make_ten_files())[-1]
        self.assertEqual(len(ten[2]['images']),10)
        rejected=self.run_turn('describe',[self.files[0]],ten[1],ten[2])[-1]
        self.assertEqual(len(rejected[2]['images']),10)
        self.assertEqual(rejected[0]['value']['files'],[self.files[0]])
        self.assertTrue(rejected[0]['interactive'])
        self.assertIn('exceed 10',rejected[1][-1]['content'])
    def test_eleven_uploads_rejected_before_state_changes(self):
        eleven=self.make_ten_files()+[self.files[0]]
        rejected=self.run_turn('describe all images',eleven)[-1]
        self.assertEqual(rejected[2],{})
        self.assertEqual(rejected[0]['value']['files'],eleven)
        self.assertIn('at most 10',rejected[1][-1]['content'])
    def test_three_actions(self):
        final=self.run_turn('Describe all images. Compare images 1 and 2. Draw boxes around all people in image 3.',self.files)[-1]
        self.assertEqual(final[2]['selected_ids'],[3])
        self.assertEqual(len(list(self.root.glob('results/multi_chat/*/*/turn.json'))),3)
    def test_partial_failure_log(self):
        def analyze(engine,item,*args):
            if item['id']==2: raise ValueError('test failure')
            return 'test',[],{'image_id':item['id']}
        self.ns['analyze_one']=analyze
        self.run_turn('describe all images',self.files)
        import json
        log=next(self.root.glob('results/multi_chat/*/*/turn.json'))
        self.assertEqual(json.loads(log.read_text())['status'],'partial')

if __name__ == '__main__':
    unittest.main()
