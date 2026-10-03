"""Retrieval/cache tests with deterministic embeddings; no Torch or model downloads."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

module_path = Path(__file__).resolve().parents[1] / 'src/upload_rag.py'
spec = importlib.util.spec_from_file_location('upload_rag_test_target', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RAG = module.UploadedImageRAG

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state={'session_id':'session_a','images':[]}
        for i,color in enumerate(['red','blue','green']):
            path=self.root/f'image_{i}.png'
            Image.new('RGB',(16,16),color).save(path)
            self.state['images'].append({'id':[7,2,9][i],'path':str(path),'filename':path.name})
        self.calls=[]
        def ask(path,question,**kwargs):
            self.calls.append((path,question))
            return {'answer':'Mock answer grounded in supplied image'}
        self.engine=SimpleNamespace(root=self.root,
            clip=SimpleNamespace(config=SimpleNamespace(_name_or_path='test',_commit_hash='r1',projection_dim=3)),
            clip_processor=SimpleNamespace(),ask=ask)
        self.image_mock=patch.object(RAG,'_embed_images',return_value=np.eye(3,dtype=np.float32)).start()
        self.query_mock=patch.object(RAG,'_embed_query',return_value=np.array([0,1,0],dtype=np.float32)).start()
        self.addCleanup(patch.stopall)
    def build(self):
        return RAG.from_state(self.engine,self.state)
    def test_rank_and_stable_original_ids(self):
        hits=self.build().search('blue image',10)
        self.assertEqual([x['image_id'] for x in hits],[2,7,9])
        self.assertEqual(hits[0]['similarity'],1)
    def test_cache_roundtrip_does_not_reembed(self):
        first=self.build()
        second=self.build()
        self.assertEqual(first.cache_status,'built')
        self.assertEqual(second.cache_status,'loaded')
        self.assertEqual(self.image_mock.call_count,1)
        self.assertEqual(first.search('blue'),second.search('blue'))
    def test_changed_content_rebuilds(self):
        self.build()
        Image.new('RGB',(16,16),'yellow').save(self.state['images'][0]['path'])
        self.assertEqual(self.build().cache_status,'built')
        self.assertEqual(self.image_mock.call_count,2)
    def test_changed_model_rebuilds(self):
        self.build()
        self.engine.clip.config._commit_hash='r2'
        self.assertEqual(self.build().cache_status,'built')
        self.assertEqual(self.image_mock.call_count,2)
    def test_different_session_has_separate_cache(self):
        first=self.build()
        self.state['session_id']='session_b'
        second=self.build()
        self.assertNotEqual(first.cache_path,second.cache_path)
        self.assertEqual(second.cache_status,'built')
    def test_corrupt_cache_rebuilds(self):
        index=self.build()
        index.cache_path.write_bytes(b'not an npz')
        self.assertEqual(self.build().cache_status,'built')
    def test_mutated_source_rejected_before_answer(self):
        index=self.build()
        Path(self.state['images'][0]['path']).unlink()
        with self.assertRaises(RuntimeError): index.answer('blue','what is visible?')
        self.assertEqual(self.calls,[])
    def test_answer_uses_retrieved_original_and_attaches_id(self):
        result=self.build().answer('blue','what is visible?',top_k=1)
        self.assertEqual(result['answers'][0]['image_id'],2)
        self.assertEqual(self.calls[0][0],self.state['images'][1]['path'])
        self.assertIn('exactly one retrieved image',self.calls[0][1])
        self.assertIn('not probabilities',result['note'])
    def test_invalid_queries_and_k(self):
        index=self.build()
        for query in ['',None]:
            with self.assertRaises(ValueError): index.search(query)
        for k in [0,-1,True,1.2]:
            with self.assertRaises(ValueError): index.search('blue',k)
        self.query_mock.return_value=np.zeros(3,dtype=np.float32)
        with self.assertRaises(ValueError): index.search('blue')
    def test_invalid_state_rejected(self):
        for state in [dict(self.state,session_id='../escape'),dict(self.state,images=[]),
                      dict(self.state,images=[self.state['images'][0]]*2)]:
            with self.assertRaises(ValueError): RAG.from_state(self.engine,state)
    def test_invalid_vectors_not_persisted(self):
        self.image_mock.return_value=np.zeros((3,3),dtype=np.float32)
        with self.assertRaises(ValueError): self.build()
        self.assertFalse((self.root/'results/upload_rag/session_a/index.npz').exists())

if __name__=='__main__': unittest.main()
