from agent3 import IncrementalAgent

bot = IncrementalAgent()

def agent(obs):
    return bot.predict(obs)