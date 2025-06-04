from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pprint import pprint
sys.path.append("src")

from src.graph import graph as graph_full

def main():
    topic = input("Enter research topic: ").strip()
    draft = input("Enter research draft: ").strip()

    thread = {'configurable': {'thread_id': '17'}}

    print("\n--- Generating researchers ---\n")
    initial_state = {'topic': topic, 'draft': draft}

    for event in graph_full.stream(initial_state, thread, stream_mode='values'):
        researchers = event.get('researchers', '')
        if researchers:
            for researcher in researchers:
                print(f'Researcher: {researcher.name}')
                print(f'Role: {researcher.role}')
                print(f'Knowledge Domain: {researcher.domain}')
                print('-' * 50)

    # Loop for multiple rounds of human feedback
    while True:
        feedback = input("\nEnter human feedback (or press Enter to continue, or type 'done' to finish): ").strip()
        if not feedback or feedback.lower() in ['done', 'skip']:
            break

        graph_full.update_state(thread, {'feedback': feedback}, as_node='human_feedback')
        for _ in graph_full.stream(None, thread, stream_mode='values'):
            pass  # wait until the graph halts again

        print("\n--- Updated researchers ---\n")
        researchers = graph_full.get_state(thread).values.get('researchers', [])
        for researcher in researchers:
            print(f'Researcher: {researcher.name}')
            print(f'Role: {researcher.role}')
            print(f'Knowledge Domain: {researcher.domain}')
            print('-' * 50)

    # Continue execution
    graph_full.update_state(thread, {'feedback': None}, as_node='human_feedback')
    print("\n--- Continuing the graph to completion ---\n")
    for event in graph_full.stream(None, thread, stream_mode='updates'):
        print("--Node--")
        print(next(iter(event.keys())))

    final_state = graph_full.get_state(thread)
    report = final_state.values.get('content')
    output_file = "samples/literature_review.md"
    if report:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {topic}\n\n{report}")
        print(f"Final report written to {output_file}")
    else:
        print("No report was generated.")

if __name__ == "__main__":
    main()
