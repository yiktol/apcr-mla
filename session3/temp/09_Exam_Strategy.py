
import streamlit as st
import pandas as pd
import random
from PIL import Image
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="AWS ML Engineer Associate Exam Simulator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define AWS color scheme
aws_colors = {
    "primary": "#232F3E",    # AWS Navy
    "secondary": "#FF9900",  # AWS Orange
    "accent1": "#1A2B3C",    # Dark blue
    "accent2": "#00A1C9",    # Light blue
    "text": "#16191F",       # Dark text
    "success": "#1E8E3E",    # Green for success
    "warning": "#F9CB9C",    # Light orange for warnings
    "error": "#D13212"       # Red for errors/wrong answers
}

# Custom CSS
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {aws_colors["primary"]};
    }}
    .stButton button {{
        background-color: {aws_colors["secondary"]};
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }}
    .stButton button:hover {{
        background-color: {aws_colors["primary"]};
        color: {aws_colors["secondary"]};
    }}
    .correct-answer {{
        background-color: #E6F7E6;
        padding: 1rem;
        border-left: 5px solid {aws_colors["success"]};
        border-radius: 5px;
        margin: 1rem 0;
    }}
    .wrong-answer {{
        background-color: #FFEBEE;
        padding: 1rem;
        border-left: 5px solid {aws_colors["error"]};
        border-radius: 5px;
        margin: 1rem 0;
    }}
    .info-box {{
        background-color: #E3F2FD;
        padding: 1rem;
        border-left: 5px solid {aws_colors["accent2"]};
        border-radius: 5px;
        margin: 1rem 0;
    }}
    .aws-card {{
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    .sidebar .sidebar-content {{
        background-color: {aws_colors["primary"]};
    }}
    .css-1d391kg {{
        background-color: {aws_colors["primary"]};
    }}
    footer {{
        visibility: hidden;
    }}
    .progress-container {{
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #f0f2f5;
    }}
    .stProgress > div > div > div > div {{
        background-color: {aws_colors["secondary"]};
    }}
    .mode-selector {{
        background-color: #f0f2f5;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
</style>
""", unsafe_allow_html=True)

# Function to generate AWS-style progress bar
def aws_progress_bar(percentage, text=""):
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>{text}</span>
            <span>{percentage}%</span>
        </div>
        <div style="height: 8px; width: 100%; background-color: #E6E6E6; border-radius: 4px;">
            <div style="height: 100%; width: {percentage}%; background-color: {aws_colors["secondary"]}; border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Function to create an AWS-style card
def aws_card(content, key=None):
    st.markdown(f'<div class="aws-card">{content}</div>', unsafe_allow_html=True)

# Generate AWS-style header with logo
def aws_header():
    cols = st.columns([1, 5])
    
    # Create an AWS-style logo
    
    aws_logo = """
    <svg width="100" height="60" viewBox="0 0 100 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M29.5 30C29.5 32.5 27.5 39 23 39C18.5 39 16.5 33 16.5 30C16.5 27 18.5 21 23 21C27.5 21 29.5 27.5 29.5 30Z" fill="#FF9900"/>
        <path d="M55.5 30C55.5 32.5 53.5 39 49 39C44.5 39 42.5 33 42.5 30C42.5 27 44.5 21 49 21C53.5 21 55.5 27.5 55.5 30Z" fill="#FF9900"/>
        <path d="M80 25.5V35.5C80 37.7 78.2 39.5 76 39.5H69C66.8 39.5 65 37.7 65 35.5V25.5C65 23.3 66.8 21.5 69 21.5H76C78.2 21.5 80 23.3 80 25.5Z" fill="#FF9900"/>
    </svg>
    """
    
    with cols[0]:
        # st.markdown(aws_logo, unsafe_allow_html=True)
        st.image("https://d0.awsstatic.com/logos/powered-by-aws.png", width=200)
    
    with cols[1]:
        st.markdown("<h1 style='color: #232F3E; margin-bottom: 0;'>Machine Learning Engineer - Associate</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #FF9900; margin-top: 0;'>Certification Exam Simulator</h3>", unsafe_allow_html=True)

# Question data
questions = [
  {
    "question": "A machine learning engineer wants to optimize costs while training a large deep learning model for fraud detection on Amazon SageMaker. What is the recommended approach to handle interruptions when using Managed Spot Training?",
    "options": [
      "Set a high value for MaxWaitTimeInSeconds",
      "Enable checkpointing to save the model's state periodically",
      "Increase the number of Spot instances used for training",
      "Use on-demand instances instead of Spot instances"
    ],
    "correct": "Enable checkpointing to save the model's state periodically",
    "explanations": {
      "Set a high value for MaxWaitTimeInSeconds": "Setting a high value for MaxWaitTimeInSeconds alone does not address the issue of interruptions caused by Spot instance terminations. It only determines the maximum time SageMaker will wait for Spot instances to become available.",
      "Enable checkpointing to save the model's state periodically": "Enabling checkpointing is the recommended approach to handle interruptions during Managed Spot Training. SageMaker saves the model's state periodically, and if an interruption occurs, the training can resume from the last checkpoint, minimizing the loss of progress.",
      "Increase the number of Spot instances used for training": "Increasing the number of Spot instances used for training does not guarantee that interruptions will not occur. Spot instances can still be terminated by AWS at any time.",
      "Use on-demand instances instead of Spot instances": "Using on-demand instances would eliminate the risk of interruptions, but it would significantly increase the cost of training compared to using Spot instances, which contradicts the goal of cost optimization."
    },
    "type": "single"
  },
  {
    "question": "A machine learning engineer is working on an NLP project and wants to improve model accuracy by tuning hyperparameters. Which approach is most operationally efficient for hyperparameter tuning?",
    "options": [
      "Use Amazon SageMaker Automatic Model Tuning",
      "Manually adjust hyperparameters and run multiple training jobs",
      "Use Amazon SageMaker Debugger to monitor the training process",
      "Create an AWS Lambda function to iterate over different hyperparameter values"
    ],
    "correct": "Use Amazon SageMaker Automatic Model Tuning",
    "explanations": {
      "Use Amazon SageMaker Automatic Model Tuning": "Amazon SageMaker Automatic Model Tuning is a built-in feature that automates finding the best combination of hyperparameters using sophisticated techniques like Bayesian optimization to efficiently search the hyperparameter space.",
      "Manually adjust hyperparameters and run multiple training jobs": "Manually adjusting hyperparameters is a tedious and time-consuming process, especially for complex models with many hyperparameters. It's inefficient and may not explore the entire hyperparameter space effectively.",
      "Use Amazon SageMaker Debugger to monitor the training process": "Amazon SageMaker Debugger is for monitoring and analyzing the training process, but it doesn't directly support hyperparameter tuning. It can provide insights but doesn't automate the tuning process.",
      "Create an AWS Lambda function to iterate over different hyperparameter values": "Creating a custom Lambda function is an overly complex and inefficient approach that requires significant development effort and doesn't leverage advanced techniques used by SageMaker Automatic Model Tuning."
    },
    "type": "single"
  },
  {
    "question": "A machine learning engineer is training a binary classification model where the business outcome requires minimizing false positives. Which metric should be optimized?",
    "options": [
      "Recall",
      "Precision",
      "Accuracy",
      "RMSE"
    ],
    "correct": "Precision",
    "explanations": {
      "Recall": "Recall measures the ability of the model to identify all positive instances, which does not directly minimize false positives. A high recall could still result in many false positives.",
      "Precision": "Precision measures the proportion of true positives among all predicted positives. Optimizing for high precision will minimize false positives, which is the desired business outcome.",
      "Accuracy": "Accuracy considers both false positives and false negatives equally. Since the goal is specifically to minimize false positives, accuracy is not the ideal metric to optimize.",
      "RMSE": "RMSE (Root Mean Squared Error) is a metric used for regression problems, not classification problems like the binary classification case described in the question."
    },
    "type": "single"
  },
  {
    "question": "A machine learning model is overfitting the training data in Amazon SageMaker. Which strategy would be most effective in mitigating this overfitting?",
    "options": [
      "Increase the learning rate",
      "Decrease the batch size",
      "Add more training data",
      "Increase the regularization parameter"
    ],
    "correct": "Increase the regularization parameter",
    "explanations": {
      "Increase the learning rate": "Increasing the learning rate is generally used to speed up convergence or escape local minima, but it does not directly address overfitting.",
      "Decrease the batch size": "Decreasing the batch size can introduce more noise into the gradient estimates, which might help with generalization to some extent, but it is not the most effective strategy for mitigating overfitting.",
      "Add more training data": "While adding more training data can help the model generalize better by providing more examples, it does not directly address overfitting if the model is already complex enough to fit the current data.",
      "Increase the regularization parameter": "Increasing the regularization parameter is a direct and effective way to mitigate overfitting. By adding a stronger penalty to the model's complexity, it encourages the model to have smaller weights, making it less likely to overfit the training data."
    },
    "type": "single"
  },
  {
    "question": "Which metric should a machine learning engineer use to evaluate a regression model predicting customer lifetime value when they want to identify potential outliers or large errors?",
    "options": [
      "F1 Score",
      "R-squared",
      "Root Mean Squared Error (RMSE)",
      "Recall"
    ],
    "correct": "Root Mean Squared Error (RMSE)",
    "explanations": {
      "F1 Score": "F1 Score is typically used for classification problems, not regression. It's not suitable for identifying outliers or large errors in regression problems.",
      "R-squared": "R-squared measures how well the model fits the data but doesn't directly measure prediction error or identify outliers. It doesn't specifically address identifying large errors or outliers in predictions.",
      "Root Mean Squared Error (RMSE)": "RMSE measures the square root of the average squared differences between predicted and actual values, making it sensitive to large errors and outliers. It's particularly good at indicating the presence of large model errors.",
      "Recall": "Recall is a metric used in classification problems to measure the proportion of actual positive cases correctly identified. It's not applicable to regression problems and doesn't provide information about the magnitude of errors."
    },
    "type": "single"
  }
]


# Helper functions
def initialize_session_state():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'exam_completed' not in st.session_state:
        st.session_state.exam_completed = False
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = len(questions)
    if 'mode' not in st.session_state:
        st.session_state.mode = "Practice Mode"
    # New score tracking approach
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = {}

def reset_session():
    st.session_state.current_question = 0
    st.session_state.answers = {}
    st.session_state.submitted = False
    st.session_state.exam_completed = False
    # Reset the correct answers tracking
    st.session_state.correct_answers = {}
    # Keep the mode selection

def next_question():
    st.session_state.submitted = False
    st.session_state.current_question += 1
    if st.session_state.current_question >= st.session_state.total_questions:
        st.session_state.exam_completed = True
        calculate_results()

def prev_question():
    st.session_state.submitted = False
    st.session_state.current_question -= 1
    if st.session_state.current_question < 0:
        st.session_state.current_question = 0

def submit_answer():
    st.session_state.submitted = True
    
    # Check if answer is correct
    if st.session_state.current_question in st.session_state.answers:
        current_q = questions[st.session_state.current_question]
        selected_answer = st.session_state.answers[st.session_state.current_question]
        
        # For single-answer questions
        if current_q["type"] == "single":
            is_correct = selected_answer == current_q["correct"]
        # For multi-select questions
        else:
            is_correct = set(selected_answer) == set(current_q["correct"])
            
        # Store whether this question was answered correctly
        st.session_state.correct_answers[st.session_state.current_question] = is_correct

def calculate_results():
    # Count correct answers
    correct_count = sum(1 for is_correct in st.session_state.correct_answers.values() if is_correct)
    percentage = (correct_count / st.session_state.total_questions) * 100
    
    # Store results in session state
    st.session_state.correct_count = correct_count
    st.session_state.percentage = percentage

def go_to_results():
    st.session_state.exam_completed = True
    calculate_results()

def set_mode(mode):
    st.session_state.mode = mode
    # Reset the test when switching modes
    reset_session()

# Charts and visualizations
def create_pie_chart():
    # Calculate correct answers for visualization
    correct_count = sum(1 for is_correct in st.session_state.correct_answers.values() if is_correct)
    total = st.session_state.total_questions
    incorrect = total - correct_count
    
    # Generate a simple bar chart as a base64 string
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Correct', 'Incorrect'], [correct_count, incorrect], color=[aws_colors["success"], aws_colors["error"]])
    plt.title('Exam Results')
    plt.ylabel('Number of Questions')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate([correct_count, incorrect]):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Return the base64 encoded image
    return base64.b64encode(buf.read()).decode()




# Main application
def main():
    initialize_session_state()
    
    # Initialize content area
    aws_header()
    
    # Sidebar with navigation and session management
    with st.sidebar:
        st.markdown(f"<h3 style='color: {aws_colors['secondary']}'>Exam Navigation</h3>", unsafe_allow_html=True)
        
        # Mode selection
        st.markdown("<div class='mode-selector'>", unsafe_allow_html=True)
        st.markdown("### Mode Selection")
        selected_mode = st.radio(
            "Choose exam mode:",
            ["Practice Mode", "Review Mode"],
            index=0 if st.session_state.mode == "Practice Mode" else 1,
            key="mode_selector"
        )
        
        # Apply mode change if needed
        if selected_mode != st.session_state.mode:
            set_mode(selected_mode)
        
        # Explain the difference between modes
        if selected_mode == "Practice Mode":
            st.markdown("In **Practice Mode**, you'll answer questions one by one and get immediate feedback.")
        else:
            st.markdown("In **Review Mode**, you can see all explanations for each question to study the material.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Progress
        st.markdown("### Progress")
        progress_percentage = int((st.session_state.current_question + 1) / st.session_state.total_questions * 100) if not st.session_state.exam_completed else 100
        aws_progress_bar(progress_percentage, f"Question {st.session_state.current_question + 1} of {st.session_state.total_questions}")
        
        # Show current score in sidebar
        correct_count = sum(1 for is_correct in st.session_state.correct_answers.values() if is_correct)
        answered_count = len(st.session_state.answers)
        
        st.markdown(f"### Current Score")
        st.markdown(f"**Correct:** {correct_count}/{answered_count} answered questions")
        if answered_count > 0:
            score_percentage = int((correct_count / answered_count) * 100)
            aws_progress_bar(score_percentage, f"Score: {score_percentage}%")
        
        # Session management
        st.markdown("### Session Management")
        if st.button("Reset Progress"):
            reset_session()
            st.rerun()
            
        st.markdown("""---""")
        
        # Add information about the exam
        st.markdown("### About This Exam")
        st.markdown("""
        This simulator contains practice questions for the AWS Machine Learning Engineer Associate Certification.
        
        **Exam domains:**
        - Data Preparation for ML
        - ML Model Development
        - Deployment and Orchestration
        - Monitoring, Maintenance, and Security
        """)
        
        st.markdown("""---""")
        st.markdown("### Resources")
        st.markdown("[AWS Documentation](https://docs.aws.amazon.com/sagemaker/)")
        st.markdown("[AWS Certification](https://aws.amazon.com/certification/)")
    
    # Display exam completed page or question page
    if st.session_state.exam_completed:
        show_results_page()
    else:
        show_question_page()
        
def show_question_page():
    current_q = questions[st.session_state.current_question]
    
    with st.container():
        st.markdown(f"## Question {st.session_state.current_question + 1}")
        st.markdown(f"<div class='aws-card'>{current_q['question']}</div>", unsafe_allow_html=True)
        
        # Show appropriate question type input (radio for single, checkbox for multiple)
        if current_q["type"] == "single":
            # Get previously selected answer if it exists
            default_value = st.session_state.answers.get(st.session_state.current_question, None)
            
            selected_option = st.radio(
                "Select one answer:",
                current_q["options"],
                index=current_q["options"].index(default_value) if default_value in current_q["options"] else 0,
                key=f"q{st.session_state.current_question}",
                on_change=None
            )
            
            # Update answers when option is selected
            st.session_state.answers[st.session_state.current_question] = selected_option
            
        else:  # Multi-select question
            selected_options = st.multiselect(
                "Select all that apply:",
                current_q["options"],
                default=st.session_state.answers.get(st.session_state.current_question, []),
                key=f"q{st.session_state.current_question}"
            )
            
            # Update answers when options are selected
            st.session_state.answers[st.session_state.current_question] = selected_options
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.current_question > 0:
                if st.button("← Previous"):
                    prev_question()
                    st.rerun()
        
        with col2:
            # In Review Mode, always show the explanation
            # In Practice Mode, require submission
            if st.session_state.mode == "Practice Mode":
                if st.button("Submit Answer"):
                    submit_answer()
                    st.rerun()
            else:
                # In Review Mode, automatically show explanations
                st.session_state.submitted = True
        
        with col3:
            if st.session_state.current_question < st.session_state.total_questions - 1:
                if st.button("Next →"):
                    next_question()
                    st.rerun()
            else:
                if st.button("See Results"):
                    go_to_results()
                    st.rerun()
    
    # Display explanation after submission or always in Review Mode
    if st.session_state.submitted or st.session_state.mode == "Review Mode":
        selected_answer = st.session_state.answers.get(st.session_state.current_question, None)
        correct_answer = current_q["correct"]
        
        st.markdown("---")
        
        # For single-select questions
        if current_q["type"] == "single":
            if selected_answer == correct_answer and st.session_state.mode == "Practice Mode":
                st.markdown(f"""
                <div class="correct-answer">
                    <h3>✅ Correct!</h3>
                    <p><strong>You selected:</strong> {selected_answer}</p>
                    <p><strong>Explanation:</strong> {current_q["explanations"][selected_answer]}</p>
                </div>
                """, unsafe_allow_html=True)
            elif selected_answer and selected_answer != correct_answer and st.session_state.mode == "Practice Mode":
                st.markdown(f"""
                <div class="wrong-answer">
                    <h3>❌ Incorrect</h3>
                    <p><strong>You selected:</strong> {selected_answer}</p>
                    <p><strong>Explanation:</strong> {current_q["explanations"][selected_answer]}</p>
                </div>
                
                <div class="correct-answer">
                    <h3>The correct answer is: {correct_answer}</h3>
                    <p><strong>Explanation:</strong> {current_q["explanations"][correct_answer]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Review Mode - show all explanations
                st.markdown(f"""
                <div class="correct-answer">
                    <h3>The correct answer is: {correct_answer}</h3>
                    <p><strong>Explanation:</strong> {current_q["explanations"][correct_answer]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # If in review mode, show explanations for all options
                if st.session_state.mode == "Review Mode":
                    st.markdown("### All Answer Explanations:")
                    for option in current_q["options"]:
                        if option != correct_answer:
                            st.markdown(f"""
                            <div class="info-box">
                                <h4>{option}</h4>
                                <p>{current_q["explanations"][option]}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Additional info if available
        if "additional_info" in current_q:
            st.markdown(f"""
            <div class="info-box">
                <h3>Additional Information</h3>
                {current_q["additional_info"]}
            </div>
            """, unsafe_allow_html=True)
        

def show_results_page():
    st.markdown("## Exam Results")
    
    # Calculate final score
    correct_count = sum(1 for is_correct in st.session_state.correct_answers.values() if is_correct)
    percentage = (correct_count / st.session_state.total_questions) * 100
    
    # Different message based on score
    if percentage >= 80:
        result_message = "Congratulations! You're well prepared for the AWS ML Engineer Associate exam!"
        result_color = aws_colors["success"]
        result_icon = "🎉"
    elif percentage >= 60:
        result_message = "Good job! With some additional study, you'll be ready for the exam."
        result_color = aws_colors["secondary"]
        result_icon = "👍"
    else:
        result_message = "Keep studying! Review the areas where you had difficulty."
        result_color = aws_colors["error"]
        result_icon = "📚"
    
    # Display results card
    st.markdown(f"""
    <div class="aws-card" style="text-align: center;">
        <h1 style="font-size: 3rem;">{result_icon}</h1>
        <h2>Your Score: {correct_count}/{st.session_state.total_questions} ({percentage:.1f}%)</h2>
        <p style="color: {result_color}; font-weight: bold; font-size: 1.2rem;">{result_message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display results visualization
    col1, col2 = st.columns([2, 3])
    
    with col1:
        chart_img = create_pie_chart()
        st.markdown(f"""
        <div class="aws-card">
            <h3 style="text-align: center;">Performance Summary</h3>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{chart_img}" style="max-width: 100%;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="aws-card">
            <h3>Recommended Next Steps</h3>
            <ol>
                <li>{"Review your incorrect answers and study those topics" if correct_count < st.session_state.total_questions else "You got everything correct! Great job!"}</li>
                <li>Complete the recommended AWS Skill Builder Training Path</li>
                <li>Try additional practice questions to reinforce your knowledge</li>
                <li>{"Schedule your certification exam when you're consistently scoring above 80%" if percentage < 80 else "You're ready to schedule your certification exam!"}</li>
            </ol>
            <h3>Key Resources</h3>
            <ul>
                <li><a href="https://aws.amazon.com/certification/certified-machine-learning-specialty/">AWS ML Engineer Certification Page</a></li>
                <li><a href="https://aws.amazon.com/sagemaker/">Amazon SageMaker Documentation</a></li>
                <li><a href="https://docs.aws.amazon.com/machine-learning">AWS Machine Learning Documentation</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Show which questions were answered correctly/incorrectly
    st.markdown("## Question Summary")
    
    summary_rows = []
    for i, q in enumerate(questions):
        is_correct = st.session_state.correct_answers.get(i, False)
        status = "✅ Correct" if is_correct else "❌ Incorrect"
        summary_rows.append([i+1, q['question'][:80] + "...", status])
    
    summary_df = pd.DataFrame(summary_rows, columns=["#", "Question", "Status"])
    
    # Style the dataframe
    def highlight_status(val):
        if val == "✅ Correct":
            return f"background-color: {aws_colors['success']}33; color: {aws_colors['success']}"
        elif val == "❌ Incorrect":
            return f"background-color: {aws_colors['error']}33; color: {aws_colors['error']}"
        return ""
    
    styled_summary = summary_df.style.applymap(highlight_status, subset=["Status"])
    st.dataframe(styled_summary, use_container_width=True)
    
    # Buttons for navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Restart Exam"):
            reset_session()
            st.rerun()
    
    with col2:
        if st.button("Review Questions"):
            st.session_state.current_question = 0
            st.session_state.exam_completed = False
            st.session_state.mode = "Review Mode"
            st.rerun()

if __name__ == "__main__":
    main()
