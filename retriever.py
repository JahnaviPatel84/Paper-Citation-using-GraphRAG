from clients import client
# 6. Community Summaries → Community Answers → Global Answer
def generate_answers_from_communities(community_summaries, query):
    intermediate_answers = []
    for summary in community_summaries:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer the following query based on the provided summary."},
                {"role": "user", "content": f"Query: {query} Summary: {summary}"}
            ]
        )
        intermediate_answers.append(response.choices[0].message.content)

    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Combine these answers into a final, concise response."},
            {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
        ]
    )
    final_answer = final_response.choices[0].message.content
    return final_answer
