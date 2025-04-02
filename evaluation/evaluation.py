from ragas import evaluate
from datasets import Dataset
from ragas.metrics import context_recall, faithfulness, answer_relevancy
from frontend.query_handler import call_rag_app
import os, json, logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

questions = ["What is the habit loop, and how does it work?",
             "How does the book explain the role of identity in habit formation?",
             "What role does environment play in shaping habits?",
             "How can habits improve productivity?",
             "Who is the author of \"Atomic Habits,\" and what is the book about?",
             "What are the four laws of behavior change mentioned in the book?",
             "How can the book's principles be applied to learning new skills?",
             "What is the difference between outcome-based habits and identity-based habits?",
             "What does the book say about the compounding effect of habits over time?",
             "Do you agree with the idea that small habits can lead to significant changes? Why or why not?",
             "If someone wants to quit smoking, how can they apply the four laws of behavior change?",
             "What is the \"2-min rule\" and how does it help?",
             "What is the main ideaa of the book \"Atomic Habits\"? ",
             "What examples from the book resonate most with you?",
             "How would you summarize the book in one sentence?",
             "How has \"Atomic Habits\" influenced your approach to personal growth?",
            ]
reference = ["The habit loop consists of four stages: cue, craving, response, and reward. It is a feedback loop that helps create automatic habits by associating rewards with specific cues.",
            "Identity-based habits focus on who you wish to become rather than what you want to achieve. Every action you take reinforces the type of person you want to be, and habits are a way to align your actions with your desired identity.",
            "The environment shapes habits by providing cues that trigger behaviors. By designing your environment to make good habits easier and bad habits harder, you can influence your behavior effectively.",
            "Habits improve productivity by automating repetitive tasks, reducing decision fatigue, and creating consistent routines that optimize time and focus.",
            "The author of \"Atomic Habits\" is James Clear. The book focuses on the power of small habits and how they can lead to significant changes in personal and professional life.",
            "The four laws of behavior change are: Make it obvious, Make it attractive, Make it easy, and Make it satisfying. These laws help in creating good habits and breaking bad ones.",
            "The principles of the book can be applied to learning new skills by focusing on small, incremental improvements and creating a system that supports consistent practice.",
            "Outcome-based habits focus on the results you want to achieve, while identity-based habits focus on the type of person you want to become. Identity-based habits are more effective for long-term change.",
            "The book explains that habits compound over time, meaning that small changes can lead to significant results when consistently applied. This compounding effect can be seen in various aspects of life, including health, productivity, and relationships.",
            "The book \"Atomic Habits\" emphasizes that small, consistent changes can compound over time to create significant transformations, highlighting the power of incremental progress.",
            "The four laws of behavior change can be applied to quit smoking by removing cues, reducing the attractiveness of smoking, creating barriers to the habit, and associating it with negative outcomes while rewarding progress.",
            "The \"2-min rule\" suggests that when starting a new habit, it should take less than two minutes to do. This helps in overcoming procrastination and makes it easier to start new habits.",
            "The main idea of the book \"Atomic Habits\" is that small, consistent changes can lead to significant improvements in personal and professional life.",
            "The examples that resonate most with me are the stories of individuals who transformed their lives through small habit changes, such as the British Cycling team and the author himself.",
            "The book can be summarized as a guide to building good habits and breaking bad ones through small, incremental changes.",
            "\"Atomic Habits\" has influenced my approach to personal growth by emphasizing the importance of small, consistent changes and the power of identity in shaping behavior.",
            ]
answers = []
contexts = []

# Inference
for query in questions:
    response_data = call_rag_app(query, OPENAI_API_KEY)
    answers.append(response_data["response"])
    contexts.append(response_data["retrieved_documents"])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "reference": reference
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Evaluate the pipeline
results = evaluate(
    dataset=dataset,
    metrics=[context_recall, faithfulness, answer_relevancy],
)

# Print evaluation results
print("Evaluation Results:")
df = results.to_pandas()
df.to_csv("evaluation_results.csv", index=False)
print(df)