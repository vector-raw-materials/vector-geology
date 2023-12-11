Contributing
============

We welcome and encourage everyone to contribute to vector-geology! Contributions can be questions, bug reports, feature requests, and new code. Here is how to get started.

Issues
------

Questions
^^^^^^^^^

For questions about vector-geology (e.g., its applications, functionality, and usage), please search the existing issues for related questions. If your question has not already been asked, then make a new issue.

Reporting Bugs
^^^^^^^^^^^^^^

Please report bugs on the issue page  and label the issue as a bug. The template asks essential questions for you to answer so that we can understand, reproduce, and fix the bug. Be verbose! Whenever possible, provide tracebacks and/or error messages, screenshots, and sample code or other files.

Feature Requests
^^^^^^^^^^^^^^^^

We encourage users to submit ideas for improvements to the vector-geology project. For this please create an issue on the `issue page with the *Feature Request* template and label. Please make sure to use a descriptive title and to provide ample background information to help us implement that functionality in the future.

Contributing New Code
---------------------

Any code contributions are welcome, whether fixing a typo or bug, adding new post-processing/plotting functionality, improving core functionality, or anything that you think should be in the repository.

Contributions should address an open issue (either a bug or a feature request). If you have found a new bug or have an idea or a new feature, then please open the issue for discussion and link to that issue in your pull request.

Python Code Guidelines
^^^^^^^^^^^^^^^^^^^^^^

We aim to follow particular Python coding guidelines to improve the sustainability and positive impact of this community project:

- Follow `The Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_, most importantly "readability counts" when writing Python code.
- Adhere to the `Style Guide for Python Code (PEP8) <https://www.python.org/dev/peps/pep-0008/>`_.
- Write thorough and effective documentation: Make a docstring for each module, function, class, and method, all following `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_ for high-level guidelines and `Google Python Style Guidelines <http://google.github.io/styleguide/pyguide.html>`_ for Syntax.

Example function documentation::

    def func(arg1: int, arg2: float) -> int:
        """A concise one-line summary of the function.

        Additional information and description of the function, if necessary. This
        can be as long and verbose as you think is necessary for other users and
        developers to understand your functionality.

        Args:
            arg1 (int): Description of the first argument.
            arg2 (float): Description of the second argument. Please use hanging
                indentation for multi-line argument descriptions.

        Returns:
            (int) Description of the return value(s)
        """
        return 42

- The code should explain the *what* and *how*. Add inline comments to explain the *why*. If an inline comment seems to be needed, consider first making the code more readable. For all comments, follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
- Test every line of code. Untested code is dead code.

Licensing
^^^^^^^^^

All contributed code will be licensed under `a EUPL-1.2 license <https://github.com/vector-raw-materials/vector-geology/blob/main/LICENSE>`_. If you did not write the code yourself, it is your responsibility to ensure that the existing license is compatible and included in the contributed files. In general, we discourage contributing third-party code.
