def after_preprocess(func):
    """Decorator to ensure that self.preprocess() is called before the function is executed."""
    def wrapper(self, *args, **kwargs):
        if not self.preprocessed:
            self.preprocess()
        return func(self, *args, **kwargs)
    return wrapper