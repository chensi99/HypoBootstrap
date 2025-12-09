class Task:

    def __new__(cls, *, task_name: str, **kwargs):
        """
            Instantiate different tasks based on `task_name`.
        """

        match task_name:
            case "list_function":
                from .list_function import ListFunctionTask
                instance = super().__new__(ListFunctionTask)
            case "arc":
                from .arc import ArcTask
                instance = super().__new__(ArcTask)
            case "acre":
                from .acre import AcreTask
                instance = super().__new__(AcreTask)
            case "scan":
                from .scan import ScanTask
                instance = super().__new__(ScanTask)
            case _:
                raise ValueError(f"Unknown task: {task_name}")
        
        instance.__init__(task_name=task_name, **kwargs)

        required_member_variables = [
            "name",
            "idx",
            "train_examples",
            "test_examples",
            "train_examples_str",
            "test_examples_str"
        ]
        for var in required_member_variables:
            if not hasattr(instance, var) or getattr(instance, var) is None:
                raise TypeError(f"Missing required member variable '{var}' in class '{type(instance)}'")

        return instance
    
    def __init__(self, *, task_name: str, examples: dict):
        self.name: str = task_name
