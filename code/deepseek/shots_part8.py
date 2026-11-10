"""Paper screenshots for Part 8 (the specialised models). Run: python shots_part8.py  (or ONLY=name1,name2)"""
import os
from paper_shots import shot

JOBS = [
    # DeepSeek-Coder (2401.14196)
    dict(name='p8-coder-deps', arxiv_id='2401.14196', page=3, anchor='will consider how to leverage the dependencies',
         highlight=['arrange these files in an order that ensures the context each file relies on is placed before that file'], above=4, below=48),
    dict(name='p8-coder-fim', arxiv_id='2401.14196', page=7, anchor='code completion abilities. Moreover',
         highlight=['we ultimately choose the 50% PSM rate as our preferred training policy'], above=4, below=120),
    # DeepSeek-Coder-V2 (2406.11931)
    dict(name='p8-coderv2-ckpt', arxiv_id='2406.11931', page=4, anchor='Both DeepSeek-Coder-V2 and DeepSeek-Coder-V2-Lite',
         highlight=['we continue the pre-training process from an intermediate checkpoint of DeepSeek-V2',
                    'exposed to a total of 10.2T high-quality tokens'], above=4, below=60),
    # DeepSeekMath (2402.03300)
    dict(name='p8-math-pipeline', arxiv_id='2402.03300', page=4, anchor='Figure 2 | An iterative pipeline',
         highlight=['An iterative pipeline that collects mathematical web pages from Common Crawl'], figure=True, below=0),
    dict(name='p8-math-iterations', arxiv_id='2402.03300', page=4, anchor='Domains where over 10% of the web pages',
         highlight=['Domains where over 10% of the web pages have been collected are classified as math-related',
                    'After four iterations of data collection, we end up with 35.5M mathematical web pages, totaling 120B',
                    'nearly 98% of the data has already been collected in the third iteration'], above=4, below=130),
    dict(name='p8-math-grpo', arxiv_id='2402.03300', page=12, anchor='Figure 4 | Demonstration of PPO and our GRPO',
         highlight=['GRPO foregoes the value model', 'estimating the baseline from group scores'], figure=True, below=16),
    dict(name='p8-math-majk', arxiv_id='2402.03300', page=20, anchor='Figure 7 | The Maj@K and Pass@K',
         highlight=['RL enhances Maj@K but not Pass@K'], figure=True, below=0),
    dict(name='p8-math-why-rl', arxiv_id='2402.03300', page=20, anchor='In this paper, we conduct reinforcement learning',
         highlight=['rendering the output distribution more robust',
                    'boosting the correct response from TopK rather than the enhancement of fundamental capabilities'],
         above=4, below=100),
    # DeepSeekMath-V2 (2511.22570)
    dict(name='p8-mathv2-meta', arxiv_id='2511.22570', page=2, anchor='The approach described in Section 2.1.1',
         highlight=['the verifier can receive full reward by predicting the correct scores while hallucinating non-existent issues',
                    'we introduce meta-verification'], above=4, below=100),
    dict(name='p8-mathv2-putnam', arxiv_id='2511.22570', page=7, anchor='To validate our results, mathematical experts',
         highlight=['scoring 118/120 and surpassing the highest human score of 90',
                    'the hardest IMO-level problems remain challenging for our model'], above=4, below=140, column='full'),
    # DeepSeek-Prover-V1.5 (2408.08152)
    dict(name='p8-proverv15-rl', arxiv_id='2408.08152', page=3, anchor='Reinforcement Learning: We employ the GRPO',
         highlight=['reinforcement learning from proof assistant feedback (RLPAF)',
                    'Verification results from the Lean prover serve as reward supervision',
                    'By assigning intrinsic rewards'], above=4, below=175),
    # DeepSeek-Prover-V2 (2504.21801)
    dict(name='p8-proverv2-cold', arxiv_id='2504.21801', page=2, anchor='Figure 2 | Overview of the cold-start data',
         highlight=['with sorry placeholders for omitted proof details', 'A 7B prover model then recursively solves the decomposed subgoals'],
         figure=True, below=76),
    dict(name='p8-proverv2-putnam', arxiv_id='2504.21801', page=9, anchor='We evaluate our model on the latest release of PutnamBench',
         highlight=['Our initial run solved 49 problems', 'we excluded two problems due to misformulated statements',
                    'The final result stands at 47 problems successfully solved'], above=4, below=95),
    dict(name='p8-proverv2-hack', arxiv_id='2504.21801', page=9, anchor='Reward Hacking in Reinforcement Learning',
         highlight=['traced to a user interface bug in Lean 4.9.0', 'to exploit this user-interface bug'], above=4, below=110),
    # DeepSeek-VL2 (2412.10302)
    dict(name='p8-vl2-tiling', arxiv_id='2412.10302', page=3, anchor='Inspired by recent advances in VLMs',
         highlight=['We select the resolution', 'that minimizes the padding area', 'plus one global thumbnail tile',
                    '729 visual embeddings of 1152 dimensions per tile'], above=4, below=140),
    # Janus-Pro (2501.17811)
    dict(name='p8-januspro-arch', arxiv_id='2501.17811', page=2, anchor='Figure 3 | Architecture of our Janus-Pro',
         highlight=['We decouple visual encoding for multimodal under'], figure=True, below=0),
    dict(name='p8-ocr-arch', arxiv_id='2510.18234', page=4, anchor='Figure 3 | The architecture of DeepSeek-OCR',
         highlight=['a 16× token compressor that bridges between them'], figure=True, below=0),
    dict(name='p8-ocr-table2', arxiv_id='2510.18234', page=9, anchor='Table 2 | We test',
         highlight=['96.5%', '10.5×', '59.1%', '19.7×'], above=4, below=265),
    dict(name='p8-ocr-news', arxiv_id='2510.18234', page=11, anchor='As shown in Table 4, some categories',
         highlight=['slides which only need 64 vision tokens', 'the text tokens in newspapers are 4-5,000'], above=4, below=150),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
