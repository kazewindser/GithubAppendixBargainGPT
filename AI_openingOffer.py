from openai import OpenAI
from pydantic import BaseModel, Field
import os
import csv

client = OpenAI(api_key=os.environ.get('AI_Bargaining_KEY'))

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

user_prompt_P1_stage1_initial = """
Follow the game rules defined in the system prompt.
-Role: P1
-The human player is P2.
-Current Stage: 1
-You must propose how many points (0–100) to give to P2：Offer_to_P2_stage1: <integer>
-and describe the reason why you made this proposal：<string>
"""

MODEL = "gpt-5.4"
TEMP = 1

class P1_stage1_initial(BaseModel):
    # AI作为p1一开始的offer
    Offer_to_P2_stage1: int = Field(ge=0, le=100)
    Reason: str

results = []

for i in range(130):
    response = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system",
             "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": user_prompt_P1_stage1_initial}
        ],
        temperature=TEMP,
        text_format=P1_stage1_initial,
    )
    response_content = response.output_parsed
    prediction_dict = response_content.model_dump()
    results.append({
        'trial': i + 1,
        'Offer_to_P2_stage1': prediction_dict['Offer_to_P2_stage1'],
        'Reason': prediction_dict['Reason']
    })

# Save to CSV
csv_filename = 'AI_opening_offers2.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['trial', 'Offer_to_P2_stage1', 'Reason']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {csv_filename}")










