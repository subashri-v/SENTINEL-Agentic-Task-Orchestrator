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
     "AttractionandRecruitmentprocessisapplicabletoalllevelsofELSA"),
    (HR, "What are the parts of the process that keeps active members motivated and coming back?",
     "a.Accommodation;b.Motivation;c.Engagement;andd.Retention"),
    (HR, "For which period was the international association's people-management strategy drawn up?",
     "HumanResourcesStrategyofELSAInternational2024-2026istheproductofextensiveefforts"),
    (HR, "What is the training and development process meant to create among members?",
     "senseofsolidarityandunitywithinELSAandamongitsmembers"),
    (HR, "What kind of weekend event can a national group hold with its local groups when new officers take over?",
     "NationalJointTransitionWeekend"),
    (HR, "How can a group ask its volunteers about their working environment to keep them for another term?",
     "Getmembers’feedbackabouttheenvironmentandworkethic"),
    # Attention Is All You Need
    (ATT, "How well did the model do at English-to-French translation and how long was it trained?",
     "BLEUscoreof41.0aftertrainingfor3.5days"),
    (ATT, "What stops a decoder position from looking at tokens that come after it?",
     "Thismasking,combinedwithfactthattheoutputembeddings"),
    (ATT, "How does the network learn where each word sits in the sequence?",
     "injectsomeinformationabouttherelativeorabsolutepositionofthetokens"),
    (ATT, "What regularisation rate was chosen for the sub-layer outputs in the base configuration?",
     "Pdrop=0.1"),
    (ATT, "What is the effect on results of using far more attention heads than optimal?",
     "qualityalsodropsoffwithtoomanyheads"),
    (ATT, "Why do the products of queries and keys get large when the dimension grows?",
     "hasmean0andvariancedk"),
    # short stories
    (STO, "Which Latin American author, famous for a novel about a century of solitude, received a Nobel in 1982?",
     "MarquezwontheNobelPrizeinLiterature"),
    (STO, "What did the dead master of the house require of the woman in exchange for part of his estate?",
     "ontheconditionthatshecontinuedreamingforthefamily"),
    (STO, "What piece of jewellery identified the woman on the ship?",
     "thesnakeringonherindexfinger"),
    (STO, "What did the old fortune-teller say she had seen in her sleep about the man?",
     "Idreamedhewasdreamingaboutme"),
    (STO, "For what purpose did the young woman travel to Austria between the wars?",
     "hadcometoAustriabetweenthewars"),
    (STO, "In the writing-technique exercise, what happened to the vehicles along the seafront?",
     "hugewavepickingupseveralcars"),
    # GRAPHite paper
    (GRA, "How many repeated noisy predictions are averaged at test time to estimate uncertainty?",
     "T=30stochasticforwardpasses"),
    (GRA, "How much better does the model score when edges linking shared state variables are added?",
     "contributea+0.509F1improvement"),
    (GRA, "How large is the hand-verified test benchmark in contracts and functions?",
     "121contracts,1,392functions"),
    (GRA, "What share of skipped training contracts could be salvaged by handling compiler versions?",
     "approximately14%ofcurrently-rejectedWildcontractscouldberecovered"),
    (GRA, "How many neighbourhood hops does the attribution method examine around each flagged function?",
     "withk=2hopsoneveryflaggedfunction"),
    (GRA, "For how many epochs is the best model further trained on the curated subset?",
     "furthertrainedfor15epochs"),
]

header = ("# Labeled retrieval cases. Lines starting with '#' are ignored.\n"
          "# expected_text is matched with whitespace removed. Regenerate with: python evals/make_retrieval_cases.py\n")
out = Path(__file__).parent / "retrieval_cases.jsonl"
out.write_text(header + "\n".join(
    json.dumps({"question": q, "expected_source": s, "expected_text": t}, ensure_ascii=False) for s, q, t in CASES) + "\n",
    encoding="utf-8")
print(f"wrote {len(CASES)} cases to {out}")
