from workflow import PredictionWorkflow

def main(): 
    """
    Main function to execute the prediction workflow.

    This function serves as the entry point for the heart disease prediction application.
    It instantiates the PredictionWorkflow class, which encapsulates all steps required
    for data preprocessing, model inference, and result handling. By calling the run()
    method, the workflow is executed in a modular and organized manner, promoting
    maintainability and scalability of the project.
    """
    workflow = PredictionWorkflow()  # Create an instance of the workflow manager
    workflow.run()                   # Execute the complete prediction pipeline


if __name__ == '__main__':
    # This conditional ensures that the main() function is only executed when this script
    # is run directly, and not when it is imported as a module in another script.
    main()
