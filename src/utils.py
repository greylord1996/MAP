def singleton(class_):
    """A simple decorative function for creating singletons."""
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        else:
            raise Exception(
                'The class ' + class_.__class__.__name__ + ' is a singleton!'
                + 'You tried to create the second instance of this class.'
            )
        return instances[class_]
    return getinstance

