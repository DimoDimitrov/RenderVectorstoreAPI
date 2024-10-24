import threading
from fastapi import FastAPI, HTTPException
from typing import Dict, List
from pydantic import BaseModel
import time
import logging

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    hour: int
    minute: int

class AgentCheck:
    def __init__(self):
        self.agents: Dict[str, float] = {}
        self.lock = threading.Lock()

    def check_and_register_agent(self, agent_id: str, config: AgentConfig) -> Dict[str, bool]:
        with self.lock:
            current_time = time.time()
            current_hour = time.localtime(current_time).tm_hour
            current_minute = time.localtime(current_time).tm_min

            if agent_id not in self.agents:
                self.agents[agent_id] = current_time
                return {"should_update": True}

            if current_hour == config.hour and current_minute >= config.minute:
                last_update_time = self.agents[agent_id]
                last_update_struct = time.localtime(last_update_time)
                
                if (last_update_struct.tm_hour != config.hour or 
                    last_update_struct.tm_min < config.minute or
                    last_update_struct.tm_mday != time.localtime(current_time).tm_mday):
                    self.agents[agent_id] = current_time
                    return {"should_update": True}

            return {"should_update": False}

    def delete_agent(self, agent_id: str) -> Dict[str, str]:
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                return {"message": f"Agent {agent_id} deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

agent_check = AgentCheck()

app = FastAPI()

@app.post("/check_agent")
async def check_agent(agent_id: str, config: AgentConfig):
    result = agent_check.check_and_register_agent(agent_id, config)
    return result

@app.delete("/delete_agent/{agent_id}")
async def delete_agent(agent_id: str):
    return agent_check.delete_agent(agent_id)