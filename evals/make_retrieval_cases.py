"""Regenerates evals/retrieval_cases.jsonl. Edit the list, then run: python evals/make_retrieval_cases.py

Each case: a paraphrased question, the source file, and a short phrase that must appear in the retrieved chunk
(compared with whitespace removed, because pypdf drops spaces in some PDFs).
"""
import json
from pathlib import Path

HR = "Human_Resources_Handbook.pdf"
ATT = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
STO = "short_stories.pdf"
GRA = "GRAPHite_PaperDraft.pdf"

CASES = [
    # ELSA HR handbook
    (HR, "Which two activities is the first HR process, covering bringing in new people, split into?",
     "AttractionandRecruitmentprocessisapplicabletoalllevelsofELSA", 'It is split into Attraction and Recruitment.'),
    (HR, "What are the parts of the process that keeps active members motivated and coming back?",
     "a.Accommodation;b.Motivation;c.Engagement;andd.Retention", 'Accommodation, Motivation, Engagement and Retention.'),
    (HR, "For which period was the international association's people-management strategy drawn up?",
     "HumanResourcesStrategyofELSAInternational2024-2026istheproductofextensiveefforts", '2024-2026.'),
    (HR, "What is the training and development process meant to create among members?",
     "senseofsolidarityandunitywithinELSAandamongitsmembers", 'A sense of solidarity and unity within ELSA and among its members.'),
    (HR, "What kind of weekend event can a national group hold with its local groups when new officers take over?",
     "NationalJointTransitionWeekend", 'A National Joint Transition Weekend.'),
    (HR, "How can a group ask its volunteers about their working environment to keep them for another term?",
     "Getmembers’feedbackabouttheenvironmentandworkethic", "Get members' feedback on the environment and work ethic, for example by giving days off, adjusting tools and scheduling periodic review meetings."),
    # Attention Is All You Need
    (ATT, "How well did the model do at English-to-French translation and how long was it trained?",
     "BLEUscoreof41.0aftertrainingfor3.5days", 'A BLEU score of 41.0 after training for 3.5 days on eight GPUs.'),
    (ATT, "What stops a decoder position from looking at tokens that come after it?",
     "Thismasking,combinedwithfactthattheoutputembeddings", "Masking in the decoder's self-attention, combined with the output embeddings being offset by one position, stops positions attending to later positions."),
    (ATT, "How does the network learn where each word sits in the sequence?",
     "injectsomeinformationabouttherelativeorabsolutepositionofthetokens", 'Positional encodings are added to the input embeddings to inject information about the relative or absolute position of tokens.'),
    (ATT, "What regularisation rate was chosen for the sub-layer outputs in the base configuration?",
     "Pdrop=0.1", 'Pdrop = 0.1.'),
    (ATT, "What is the effect on results of using far more attention heads than optimal?",
     "qualityalsodropsoffwithtoomanyheads", 'Quality drops off with too many heads.'),
    (ATT, "Why do the products of queries and keys get large when the dimension grows?",
     "hasmean0andvariancedk", 'If q and k components are independent with mean 0 and variance 1, their dot product has mean 0 and variance dk, so it grows with the key dimension.'),
    # short stories
    (STO, "Which Latin American author, famous for a novel about a century of solitude, received a Nobel in 1982?",
     "MarquezwontheNobelPrizeinLiterature", 'Marquez (Gabriel Garcia Marquez).'),
    (STO, "What did the dead master of the house require of the woman in exchange for part of his estate?",
     "ontheconditionthatshecontinuedreamingforthefamily", 'That she continue dreaming for the family until her dreams came to an end.'),
    (STO, "What piece of jewellery identified the woman on the ship?",
     "thesnakeringonherindexfinger", 'A snake ring on her index finger.'),
    (STO, "What did the old fortune-teller say she had seen in her sleep about the man?",
     "Idreamedhewasdreamingaboutme", 'She said she dreamed that he was dreaming about her.'),
    (STO, "For what purpose did the young woman travel to Austria between the wars?",
     "hadcometoAustriabetweenthewars", 'To study music and voice.'),
    (STO, "In the writing-technique exercise, what happened to the vehicles along the seafront?",
     "hugewavepickingupseveralcars", 'A huge wave picked up several cars.'),
    # GRAPHite paper
    (GRA, "How many repeated noisy predictions are averaged at test time to estimate uncertainty?",
     "T=30stochasticforwardpasses", 'T=30 stochastic forward passes (Monte Carlo dropout).'),
    (GRA, "How much better does the model score when edges linking shared state variables are added?",
     "contributea+0.509F1improvement", 'About +0.509 F1 (0.694 vs 0.185).'),
    (GRA, "How large is the hand-verified test benchmark in contracts and functions?",
     "121contracts,1,392functions", '121 contracts and 1,392 functions (the SmartBugs Curated benchmark).'),
    (GRA, "What share of skipped training contracts could be salvaged by handling compiler versions?",
     "approximately14%ofcurrently-rejectedWildcontractscouldberecovered", 'Approximately 14% of currently-rejected Wild contracts could be recovered.'),
    (GRA, "How many neighbourhood hops does the attribution method examine around each flagged function?",
     "withk=2hopsoneveryflaggedfunction", 'k=2 hops.'),
    (GRA, "For how many epochs is the best model further trained on the curated subset?",
     "furthertrainedfor15epochs", '15 epochs (learning rate 3e-4).'),
]

header = ("# Labeled retrieval cases. Lines starting with '#' are ignored.\n"
          "# expected_text is matched with whitespace removed. Regenerate with: python evals/make_retrieval_cases.py\n")
out = Path(__file__).parent / "retrieval_cases.jsonl"
out.write_text(header + "\n".join(
    json.dumps({"question": q, "expected_source": s, "expected_text": t, "reference": ref}, ensure_ascii=False) for s, q, t, ref in CASES) + "\n",
    encoding="utf-8")
print(f"wrote {len(CASES)} cases to {out}")


# Questions the corpus cannot answer: the RAG agent should decline instead of inventing an answer.
UNANSWERABLE = [
    "What is the capital city of Australia?",
    "Who won the 2018 FIFA World Cup?",
    "What accuracy did GRAPHite reach on the ImageNet dataset?",
    "How many layers does GPT-4 have?",
    "What is the recommended dosage of ibuprofen for adults?",
    "What BLEU score did the Transformer get on Chinese-to-English translation?",
    "Which story in the collection features a dragon?",
    "What is the annual membership fee for joining the association?",
]
unanswerable_lines = [json.dumps({"question": q}) for q in UNANSWERABLE]
(Path(__file__).parent / "unanswerable_cases.jsonl").write_text(chr(10).join(unanswerable_lines) + chr(10), encoding="utf-8")
print(f"wrote {len(UNANSWERABLE)} unanswerable cases")
