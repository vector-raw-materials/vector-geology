"""
Modified after https://github.com/pyvista/pyvista/blob/ab70c26edbcfb107286c827bd4914562056219fb/docs/make_external_gallery.py

A helper script to generate the external 2-examples gallery.
"""
import os
from io import StringIO


def format_icon(title, description, link, image):
    body = r"""
   .. grid-item-card:: {}
      :link: {}
      :text-align: center
      :class-title: pyvista-card-title

      .. image:: {}
"""
    content = body.format(title, link, image)
    return content


class Example():
    def __init__(self, title, description, link, image):
        self.title = title
        self.description = description
        self.link = link
        self.image = image

    def format(self):
        return format_icon(self.title, self.description, self.link, self.image)


###############################################################################

articles = dict(
    advanced_bayes=Example(
        title="More advanced example",
        description="Multiple Priors and likelihoods",
        link="https://gempy-project.github.io/gempy_probability/examples_basic_geology/1-thickness_problem.html#sphx-glr-examples-basic-geology-1-thickness-problem-py",
        image="https://gempy-project.github.io/gempy_probability/_images/sphx_glr_1-thickness_problem_005.png",
    ),
    simple_bayes=Example(
        title="Simple example",
        description="Build a basic Bayesian model with Pyro",
        link="https://gempy-project.github.io/gempy_probability/examples_first_example_of_inference/1-bayesian_basics/1.1_Intro_to_Bayesian_Inference.html#sphx-glr-examples-first-example-of-inference-1-bayesian-basics-1-1-intro-to-bayesian-inference-py",
        image="https://gempy-project.github.io/gempy_probability/_images/sphx_glr_1.1_Intro_to_Bayesian_Inference_004.png",
    ),
    theory=Example(
        title="Bayesian Inference Theory",
        description="A brief introduction to Bayesian inference theory.",
        link="https://gempy-project.github.io/gempy_probability/examples_intro/index.html",
        image="https://gempy-project.github.io/gempy_probability/_images/Model_space2.png",
    )
)


###############################################################################

def make_example_gallery():
    """Make the example gallery."""
    path = "./external/external_examples.rst"

    with StringIO() as new_fid:
        new_fid.write(
            """.. _external_examples:

External Examples
=================

Got an impressive Bayesian inference workflow or visualization routine? Share it with the community! Contribute your work and help others learn from your expertise. Submit a PR at `visual-bayesic/visual-bayesic <https://github.com/visual-bayesic/visual-bayesic/>`_, and we'd be excited to feature it.

.. caution::

    Please note that these 2-examples link to external websites. If any of these
    links are broken, please raise an `issue
    <https://github.com/visual-bayesic/visual-bayesic/issues>`_.

Do you have a sophisticated Bayesian inference workflow or visualization routine you would like to share? If so, please consider contributing your work by submitting a PR at `visual-bayesic/visual-bayesic <https://github.com/visual-bayesic/visual-bayesic/>`_. We welcome contributions and would be delighted to include your work in our collection.

.. grid:: 3
   :gutter: 1

"""
        )
        # Reverse to put the latest items at the top
        for example in list(articles.values())[::-1]:
            new_fid.write(example.format())

        new_fid.write(
            """

.. raw:: html

    <div class="sphx-glr-clear"></div>


"""
        )
        new_fid.seek(0)
        new_text = new_fid.read()

    # check if it's necessary to overwrite the table
    existing = ""
    if os.path.exists(path):
        with open(path) as existing_fid:
            existing = existing_fid.read()

    # write if different or does not exist
    if new_text != existing:
        with open(path, "w", encoding="utf-8") as fid:
            fid.write(new_text)

    return
