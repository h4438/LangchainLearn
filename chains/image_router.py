
from typing import Any, Dict, List, Optional, Type, cast
from langchain.schema import OutputParserException
from langchain.chains import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate
# my code





EXPLORE_CHAIN_DESC =\
        {"message": """Good for answering questions that the input is a text from a human""",\
        "content": """Good for answering questions that the input is from a book, web article, news, ..."""}

EXPLORE_ROUTER_CONST ={"template": """Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.
<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ a potentially modified version of the original input
    "content": {{content}} \\ the given content
}}}}
```
REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.
<< CANDIDATE PROMPTS >>
{destinations}
<< INPUT >>
The question: {{question}}
<< OUTPUT >>""", "inputs": ["question", "content"]}






class ExploreRouter:
    def __init__(self, chains: Dict[str, LLMChain]):
        print("Initiate Router Chain")
        self.chains = chains
        self.main_chain = None

    def build(self, model):
        # step 1: create destinations_str
        info = EXPLORE_CHAIN_DESC
        destinations = [f"{p}: {info[p]}" for p in info]
        destinations_str = "\n".join(destinations)
        # step 2: create router prompt
        template = EXPLORE_ROUTER_CONST['template']
        inputs = EXPLORE_ROUTER_CONST['inputs']
        router_template = template.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template, input_variables=inputs,\
                               output_parser=ExploreRouterOutputParser())
        # step 3: create LLMRouterChain
        router_chain = LLMRouterChain.from_llm(model, router_prompt)
        # step 4: final chain
        chain = MultiPromptChain(router_chain=router_chain, destination_chains=self.chains,\
                         default_chain=self.chains['message'], verbose=True)
        self.main_chain = chain

class ExploreRouterOutputParser(RouterOutputParser):
    # a wrapper class of RouterOutputParser for ExploreRouterChain
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            parsed = super().parse(text)
            parsed["next_inputs"]['content'] = parsed['content']
            parsed["next_inputs"]['question'] = parsed["next_inputs"]['input']
            return parsed
        except Exception as e:
            raise OutputParserException(f"Parsing text\n{text}\n raised following error:\n{e}")    

    
