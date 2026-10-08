"""Validate stream parsing without pretending to run a model server."""
import unittest
from probe_server import consume_stream
class StreamTests(unittest.TestCase):
    def test_empty_and_multitoken_chunks_and_usage(self):
        lines=[b': heartbeat\n',b'data: {"choices":[{"text":""}]}\n',
               b'data: {"choices":[{"text":"hello world"}]}\n',
               b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',b'data: [DONE]\n']
        times=iter([1.2,1.5]); r=consume_stream(lines,1.0,lambda:next(times))
        self.assertAlmostEqual(r['observed_first_text_ms'],200)
        self.assertEqual(r['text'],'hello world'); self.assertEqual(r['text_chunks'],1)
        self.assertEqual(r['usage']['completion_tokens'],2)
    def test_truncated_stream(self):
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            consume_stream([b'data: {"choices":[{"text":"partial"}]}'],0,lambda:1)
    def test_server_error(self):
        with self.assertRaisesRegex(RuntimeError,'overloaded'):
            consume_stream([b'data: {"error":"overloaded"}'],0)
    def test_no_text(self):
        with self.assertRaisesRegex(RuntimeError,'No non-empty'):
            consume_stream([b'data: [DONE]'],0,lambda:1)
if __name__=='__main__': unittest.main()
