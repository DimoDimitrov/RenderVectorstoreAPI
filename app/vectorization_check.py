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
        # Initialize as RLock to allow reentrant locking
        self._lock = threading.RLock()

    def check_and_register_agent(self, agent_id: str, config: AgentConfig) -> Dict[str, bool]:
        with self._lock:
            logger.info(f"Lock acquired - Agent {agent_id} - {config.update_type} check:")
            
            current_time = time.time()
            current_struct = time.localtime(current_time)
            
            # Get the current state before any modifications
            last_update_time = self.agents.get(agent_id)
            
            # For new agents, we need to check again after getting the lock
            # to ensure another instance hasn't registered it
            if last_update_time is None:
                # Double check if another instance registered while we were waiting
                if agent_id in self.agents:
                    last_update_time = self.agents[agent_id]
                else:
                    logger.info(f"New agent {agent_id} - registering")
                    self.agents[agent_id] = current_time
                    return {"should_update": True}
                
            last_update_struct = time.localtime(last_update_time)
            should_update = False

            # Log detailed information for every request
            logger.info(f"Agent {agent_id} - {config.update_type} check:")
            logger.info(f"  Current time: {time.strftime('%Y-%m-%d %H:%M:%S', current_struct)} (wday={current_struct.tm_wday})")
            if last_update_time is not None:
                last_update_struct = time.localtime(last_update_time)
                logger.info(f"  Last update: {time.strftime('%Y-%m-%d %H:%M:%S', last_update_struct)} (wday={last_update_struct.tm_wday})")
            else:
                logger.info(f"  Last update: None (new agent)")
            if config.update_type == "weekly":
                logger.info(f"  Config day: {config.day}")
            elif config.update_type == "hourly":
                logger.info(f"  Config hour_gap: {config.hour_gap}")

            if config.update_type == "hourly":
                hours_passed = (current_time - last_update_time) / 3600
                should_update = hours_passed >= config.hour_gap
            elif config.update_type == "daily":
                # Check if it's a different day and we haven't updated yet today
                current_day_start = time.mktime(time.struct_time((
                    current_struct.tm_year,
                    current_struct.tm_mon,
                    current_struct.tm_mday,
                    0, 0, 0,
                    current_struct.tm_wday,
                    current_struct.tm_yday,
                    current_struct.tm_isdst
                )))
                should_update = last_update_time < current_day_start
            elif config.update_type == "weekly":
                # Only allow update if:
                # 1. It's the configured day of the week
                # 2. The last update was NOT on this day
                # 3. The last update was in a previous week
                current_date = current_struct.tm_year, current_struct.tm_mon, current_struct.tm_mday
                last_update_date = last_update_struct.tm_year, last_update_struct.tm_mon, last_update_struct.tm_mday
                
                should_update = (current_struct.tm_wday == config.day and 
                               (current_date != last_update_date))

                logger.info(f"  Should update: {should_update}")

            if should_update:
                # Double-check that another instance hasn't updated while we were checking
                current_last_update = self.agents.get(agent_id)
                if current_last_update != last_update_time:
                    logger.info(f"Agent {agent_id} was updated by another instance")
                    return {"should_update": False}
                
                # Update the timestamp
                self.agents[agent_id] = current_time
                logger.info(f"Successfully updated agent {agent_id}")
                return {"should_update": True}

            return {"should_update": False}

    def delete_agent(self, agent_id: str) -> Dict[str, str]:
        with self._lock:
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