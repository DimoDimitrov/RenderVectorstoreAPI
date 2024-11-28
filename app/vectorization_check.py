from fastapi import FastAPI, HTTPException, Query
from typing import Dict
from pydantic import BaseModel
import threading
import time
import logging

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    update_type: str  # 'hourly', 'daily', or 'weekly'
    hour_gap: int = None  # For hourly updates
    hour: int = None  # For daily/weekly updates
    minute: int = None  # For daily/weekly updates
    day: int = None  # For weekly updates, 0 = Monday, 6 = Sunday

class AgentCheck:
    def __init__(self):
        self.agents: Dict[str, float] = {}
        self.lock = threading.Lock()

    def check_and_register_agent(self, agent_id: str, config: AgentConfig) -> Dict[str, bool]:
        with self.lock:
            print(f"Checking agent {agent_id} with config type:{config.update_type}")
            current_time = time.time()
            current_struct = time.localtime(current_time)

            if agent_id not in self.agents:
                self.agents[agent_id] = current_time
                return {"should_update": True}

            last_update_time = self.agents[agent_id]
            last_update_struct = time.localtime(last_update_time)

            should_update = False
            if config.update_type == "hourly":
                hours_passed = (current_time - last_update_time) / 3600  # Convert seconds to hours
                should_update = hours_passed >= config.hour_gap
            elif config.update_type == "daily":
                should_update = last_update_struct.tm_mday != current_struct.tm_mday
            elif config.update_type == "weekly":
                if (current_struct.tm_wday == config.day and
                    current_struct.tm_hour == config.hour and 
                    current_struct.tm_min >= config.minute):
                    should_update = (last_update_struct.tm_wday != config.day or
                                   last_update_struct.tm_mday != current_struct.tm_mday or
                                   last_update_struct.tm_hour < config.hour or
                                   (last_update_struct.tm_hour == config.hour and 
                                    last_update_struct.tm_min < config.minute))

            if should_update:
                self.agents[agent_id] = current_time
                print(f"Agent {agent_id} approved to update")
                return {"should_update": True}

            return {"should_update": False}

    def delete_agent(self, agent_id: str) -> Dict[str, str]:
        with self.lock:
            print(f"Deleting agent {agent_id}")
            if agent_id in self.agents:
                del self.agents[agent_id]
                return {"message": f"Agent {agent_id} deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

agent_check = AgentCheck()

app = FastAPI()

@app.post("/check_agent")
async def check_agent(agent_id: str = Query(...), config: AgentConfig = None):
    result = agent_check.check_and_register_agent(agent_id, config)
    return result

@app.delete("/delete_agent/{agent_id}")
async def delete_agent(agent_id: str):
    return agent_check.delete_agent(agent_id)