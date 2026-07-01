import pandas as pd


{
 "cells": [],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

df = pd.read_csv("data/raw/uganda_mobile_money_master.csv", nrows=5)
print(df.columns.tolist())
