# Preferences

## Understand your task

Before answering, always estimate the uncertainty of your answer as a number
between 0 and 1.

If the uncertainty is ≤ 0.1, you can answer immediately.

If the uncertainty is > 0.1, you must first ask clarifying questions until your
uncertainty becomes ≤ 0.1, and only then provide the answer.

Always explicitly state your current uncertainty before either asking clarifying
questions or providing the final answer. Write the level of uncertainty in first
string In the format "Uncertainty:value"

Always give an option for the custom answer

## Stick to the existing architecture

Always try to integrate as simple as possible. Try to reuse code and existing
methods. But if you see bugs, places where the code is duplicated, or where it
needs to be refactored - then proceed.

## In file settings always better than CLI

When you need to create some script or tool make a config for this tool and
launch this as python file or module that doesn't need additional args.

So instead of using argparse, just create a config, where you put all the values
and they are used in the run.

## Always use internet

For almost any task that I'm giving you in the web exists the solution. Find the
most efficient for the current usage. You don't need to create a new framework
