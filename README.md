# Interactive AI for Healthcare Education Platform

COMP9900 - Term 3 2026  
Project P02

## Overview

This project aims to develop an interactive web-based educational platform for Years 8-12 students to explore how artificial intelligence can be used with healthcare data.

Students will interact with simplified healthcare datasets containing different data modalities, such as:

- Clinical information
- Medical images
- Molecular or genomic information

Through guided activities, students will learn how AI models make predictions, how combining different data modalities can affect prediction results, and which features or biomarkers have the greatest influence on a prediction.

## Core Features

- Interactive browser-based educational website
- Guided AI healthcare learning activities
- Support for multiple healthcare data modalities
- Comparison of single-modality and multimodal predictions
- Feature / biomarker importance visualisation
- QR-code integration with physical workshop cards or props
- Clear instructions for students and facilitators
- Example or synthetic datasets
- Usability and functionality testing

## Repository Structure

```text
frontend/     Frontend application
backend/      Local JSON patient API and kidney model module
services/     Backend and AI-related services
data/         Example and synthetic datasets
docs/         Requirements, meetings, testing and workshop documentation
tests/        Project tests 
```

## Technology Stack

The patient preview uses React, TypeScript and Vite, with a Python standard-library HTTP API backed by JSON files. No SQL database is required for the patient pages.

See [the English setup guide](docs/JSON_PATIENT_SETUP.md) to start the backend and frontend, inspect all 40 patients, and browse each patient's CT slices.

## Team

COMP9900 - 9900-H18J-DONUT

## Client

Raymond Louie

## Status

The diabetes and kidney patient pages read the stored teaching dataset. Model training, evaluation and prediction in the frontend remain a labeled workflow demonstration.
