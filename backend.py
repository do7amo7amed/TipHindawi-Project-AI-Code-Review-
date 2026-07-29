import json
import re
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_classic.chains import LLMChain
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)

# MODEL

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = None
model = None


def load_llm():
    global tokenizer, model

    if model is not None:
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model


# TEXT GENERATION

def generate_text(prompt, max_new_tokens=2048):

    load_llm()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# VECTOR DATABASE


def build_vectorstore(text):

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(text)

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(
        documents,
        embedding,
    )

    return vectordb


def retrieve_context(query, vectordb):

    docs = vectordb.similarity_search(query, k=1)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# LLM WRAPPER

class CustomHFLLM(LLM):

    def _call(self, prompt: str, stop: Any = None):

        return generate_text(prompt)

    @property
    def _llm_type(self):

        return "custom_huggingface"


llm = CustomHFLLM()


# PROMPTS
ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["code", "context"],
    template="""
You are a senior Python code reviewer.

Use the documentation below as reference.

Documentation:
{context}

Return ONLY:

Summary:
Potential Problems:
Bugs:
Bad Practices:
"""
)

analysis_chain = LLMChain(
    llm=llm,
    prompt=ANALYSIS_PROMPT,
)

ISSUE_PROMPT = PromptTemplate(
    input_variables=["analysis", "format_instructions"],
    template="""
You are a senior Python static code analyzer.

Based on the analysis below:

{analysis}

Find REAL programming issues only.

Ignore style suggestions.

Return ONLY one valid JSON object.

{format_instructions}
"""
)

issue_chain = LLMChain(
    llm=llm,
    prompt=ISSUE_PROMPT,
)

IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["code", "issues"],
    template="""
You are a senior Python developer.

Using the following detected issues:

{issues}

Suggest improvements.

Code:

{code}

Return only

Issue:
Suggested Fix:
Improved Code:
"""
)

improvement_chain = LLMChain(
    llm=llm,
    prompt=IMPROVEMENT_PROMPT,
)


# OUTPUT PARSER

schemas = [

    ResponseSchema(
        name="file",
        description="Python filename",
    ),

    ResponseSchema(
        name="issue",
        description="Detected issue",
    ),

    ResponseSchema(
        name="severity",
        description="Low Medium High",
    ),

    ResponseSchema(
        name="explanation",
        description="Problem explanation",
    ),

    ResponseSchema(
        name="suggested_fix",
        description="Suggested fix",
    ),
]

output_parser = StructuredOutputParser.from_response_schemas(
    schemas
)

format_instructions = output_parser.get_format_instructions()


# PARSER
def extract_json_block(text):

    pattern = r"```json\s*(.*?)\s*```"

    matches = re.findall(
        pattern,
        text,
        re.DOTALL,
    )

    if not matches:
        return None

    return matches[-1].strip()


def parse_issues(text):

    json_text = extract_json_block(text)

    if json_text is None:
        return None

    try:
        return output_parser.parse(json_text)
    except Exception:
        return None

# MAIN REVIEW

def run_all(file_name, code, vectordb):

    context = retrieve_context(
        code,
        vectordb,
    )

    analysis = analysis_chain.run(
        {
            "code": code,
            "context": context,
        }
    )

    issues = issue_chain.run(
        {
            "analysis": analysis,
            "format_instructions": format_instructions,
        }
    )

    improvements = improvement_chain.run(
        {
            "code": code,
            "issues": issues,
        }
    )

    return analysis, issues, improvements