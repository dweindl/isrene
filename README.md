# Overview

`isrene` (pronounced "is-reen") is a Python package for creating dynamic models of metabolic **is**otopologue **re**action **ne**tworks.

## Features

Writing up the equations for a dynamic model of metabolic stable isotope labeling experiments can be tedious and error-prone. `isrene` aims to make this process easier by providing a Python interface for:

* specifying metabolic reaction networks in terms of reaction patterns (rule-based models) describing stoichiometry, reversibility, kinetics, and atom mappings
* specifying isotopic tracers
* conveniently specifying observation functions for mass spectrometry measurements
* exporting the rule-based model to [SBML-multi](https://doi.org/10.1515/jib-2017-0077)
* generating the full species/reaction network from the rule-based formulation and exporting it to core-[SBML](https://sbml.org/)

<div><img src="doc/gfx/overview.svg" alt="overview figure"></div>

## Installation

`isrene` is not yet available on PyPI. To install, clone this repository and run `pip install .` from the root directory.

## Getting Started

Check out the examples at `examples/` to get started.

## Documentation

So far, only example notebooks and in-source documentation are available.

## Development status

Note that `isrene` has not been thoroughly tested and is still under development. **Use with caution.**
In particular, there is insufficient error checking, and certain assumptions are made on the uniqueness of identifiers and combinations thereof.

If you are interested in a collaboration to apply or extend this package, please contact the author.
