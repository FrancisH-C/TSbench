"""Export path to be read in when running R."""
# https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure
import os

Rmodels_path = os.path.dirname(os.path.abspath(__file__))  # This is R models path
