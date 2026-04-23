import pandas as pd
from os import environ
from openai import OpenAI
from pydantic import BaseModel, Field

############# Basic Setting ##################

client = OpenAI(api_key=environ.get('AI_Bargaining_KEY'))

MODEL = "gpt-5.4"

SYSTEM_PROMPT = """
You are playing a two-player 3-stage alternating-offers bargaining game with a human user.
There are two roles: "P1" and "P2".
You will be assigned one of these roles. The human player is the other role.
The total surplus to divide is 100 points.

#Game Structure
##Stage 1：
-P1 proposes how many points to give to P2 (an integer from 0 to 100).
-P2 then decides whether to ACCEPT or REJECT.
--If P2 ACCEPTS: The game ends and the proposed allocation is implemented.
--If P2 REJECTS: The game moves to Stage 2.


##Stage 2
-P2 proposes how many points to give to P1 (an integer from 0 to 100).
-P1 then decides whether to ACCEPT or REJECT.
--If P1 ACCEPTS：The game ends and the proposed allocation is implemented.
--If P1 REJECTS: The game moves to Stage 3.


##Stage 3
-P1 proposes how many points to give to P2 (an integer from 0 to 100).
-P2 then decides whether to ACCEPT or REJECT.
--If P2 ACCEPTS: The game ends and the proposed allocation is implemented.
--If P2 REJECTS: The game ends and both players receive 0 points.


#Discounting
##P1 discount factors
-Stage 1: 1
-Stage 2: 0.6
-Stage 3: 0.36

##P2 discount factors
-Stage 1: 1
-Stage 2: 0.4
-Stage 3: 0.16


#Final Payoff
-If an agreement is reached at stage t:
-P1 payoff = (points allocated to P1) × (P1 discount factor at stage t)
-P2 payoff = (points allocated to P2) × (P2 discount factor at stage t)


**Your payoff is determined by the rules above. Try to earn as many points as possible.**
"""


class P2_stage1_stage2(BaseModel):
    # AI作为P2在stage1要不要接受或者拒绝进入stage2
    Whether_to_accept_P1_offer_stage1: bool
    Offer_to_P1_if_rejected_proceed_to_stage2: int


################################################

Trial = 20

AI_Response_P2_stage1 = []
for HumanP1toAIP2stage1 in range(101):
    for trial in range(Trial):
        ##############################################

        user_prompt_P2_stage1_stage2 = f"""
            Follow the game rules defined in the system prompt.
            -Role: P2
            -The human player is P1.
            -Current Stage: 1
            -P1 has decided to propose {HumanP1toAIP2stage1} points to you.

            -Decision required:
            1. Decide whether to ACCEPT this offer: Whether_to_accept_P1_offer_stage1: <TRUE/FALSE>
            2. If you REJECT, propose an offer for Stage 2: Offer_to_P1_if_rejected_proceed_to_stage2: <integer or -1 if accepted>
            """

        response = client.responses.parse(
            model=MODEL,
            input=[
                {"role": "system",
                 "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": user_prompt_P2_stage1_stage2}
            ],
            temperature=1,
            text_format=P2_stage1_stage2,
        )

        response_content = response.output_parsed

        # 将 Pydantic 对象转换为字典
        prediction_dict = response_content.model_dump()
        prediction_dict['Trial'] = trial + 1
        prediction_dict['HumanP1toAIP2stage1'] = HumanP1toAIP2stage1

        AI_Response_P2_stage1.append(prediction_dict)

# 输出到CSV
df = pd.DataFrame(AI_Response_P2_stage1)
df.to_csv('AI_Response_HM_openingOffer5.csv', index=False)







