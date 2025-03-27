import streamlit as st
from datasets import load_dataset

dataset = load_dataset("keeve101/common-voice-unified-splits", "zh-CN", split="train[10%:]")

print(dataset.features)
