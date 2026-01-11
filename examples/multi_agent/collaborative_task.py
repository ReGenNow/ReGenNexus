#!/usr/bin/env python3
"""
Multi-Agent Collaborative Task Example

This example demonstrates how multiple agents can collaborate to solve a task
using the ReGenNexus Core protocol. In this scenario, we have:

1. A coordinator agent that manages the overall task
2. Multiple worker agents that perform specific subtasks
3. A monitoring agent that tracks progress and reports status

Requirements:
- ReGenNexus Core installed
- Registry server running
"""

import asyncio
import logging
import random
import time
import uuid
# UAP_Client not available - this example needs updating for new API
from regennexus.core.message import Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task definitions
TASKS = [
    {"id": "task1", "name": "Data Collection", "difficulty": 3, "duration": 5},
    {"id": "task2", "name": "Data Processing", "difficulty": 5, "duration": 8},
    {"id": "task3", "name": "Analysis", "difficulty": 7, "duration": 10},
    {"id": "task4", "name": "Reporting", "difficulty": 4, "duration": 6},
    {"id": "task5", "name": "Visualization", "difficulty": 6, "duration": 7},
]

# Global task status
task_status = {task["id"]: "pending" for task in TASKS}
task_assignments = {}
task_results = {}

# Coordinator Agent
async def run_coordinator(registry_url):
    # Create a client for the coordinator
    coordinator = UAP_Client(entity_id="coordinator", registry_url=registry_url)
    
    # Connect to the registry
    await coordinator.connect()
    logger.info("Coordinator connected to registry")
    
    # Create a context for this collaboration
    collaboration_id = str(uuid.uuid4())
    logger.info(f"Created collaboration context: {collaboration_id}")
    
    # Register message handlers
    @coordinator.message_handler(intent="worker_ready")
    async def handle_worker_ready(message):
        worker_id = message.sender
        capabilities = message.payload.get("capabilities", {})
        logger.info(f"Worker {worker_id} is ready with capabilities: {capabilities}")
        
        # Find an unassigned task that matches the worker's capabilities
        assigned_task = None
        for task in TASKS:
            if (task_status[task["id"]] == "pending" and 
                task["id"] not in task_assignments.values()):
                # Check if worker has sufficient capability for this task
                if capabilities.get("skill_level", 0) >= task["difficulty"]:
                    assigned_task = task
                    break
        
        if assigned_task:
            # Assign the task to the worker
            task_status[assigned_task["id"]] = "assigned"
            task_assignments[worker_id] = assigned_task["id"]
            
            # Send task assignment to worker
            await coordinator.send_message(
                recipient=worker_id,
                intent="task_assignment",
                payload={
                    "task_id": assigned_task["id"],
                    "task_name": assigned_task["name"],
                    "duration": assigned_task["duration"],
                    "collaboration_id": collaboration_id
                }
            )
            
            logger.info(f"Assigned {assigned_task['name']} to {worker_id}")
            
            # Notify monitor about the assignment
            await coordinator.send_message(
                recipient="monitor",
                intent="task_update",
                payload={
                    "task_id": assigned_task["id"],
                    "status": "assigned",
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                    "collaboration_id": collaboration_id
                }
            )
        else:
            # No suitable tasks available
            await coordinator.send_message(
                recipient=worker_id,
                intent="no_task_available",
                payload={
                    "message": "No suitable tasks available at this time",
                    "retry_after": 5  # seconds
                }
            )
    
    @coordinator.message_handler(intent="task_completed")
    async def handle_task_completed(message):
        worker_id = message.sender
        task_id = message.payload.get("task_id")
        result = message.payload.get("result")
        
        if task_id in task_status and task_assignments.get(worker_id) == task_id:
            # Update task status
            task_status[task_id] = "completed"
            task_results[task_id] = result
            
            logger.info(f"Task {task_id} completed by {worker_id}")
            
            # Notify monitor about completion
            await coordinator.send_message(
                recipient="monitor",
                intent="task_update",
                payload={
                    "task_id": task_id,
                    "status": "completed",
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                    "result_summary": result.get("summary") if result else None,
                    "collaboration_id": collaboration_id
                }
            )
            
            # Check if all tasks are completed
            if all(status == "completed" for status in task_status.values()):
                logger.info("All tasks completed!")
                
                # Notify all workers and monitor about completion
                for worker_id in task_assignments.keys():
                    await coordinator.send_message(
                        recipient=worker_id,
                        intent="collaboration_completed",
                        payload={
                            "message": "All tasks in the collaboration have been completed",
                            "collaboration_id": collaboration_id
                        }
                    )
                
                await coordinator.send_message(
                    recipient="monitor",
                    intent="collaboration_completed",
                    payload={
                        "message": "All tasks in the collaboration have been completed",
                        "collaboration_id": collaboration_id,
                        "timestamp": time.time(),
                        "results": task_results
                    }
                )
            else:
                # Worker is available for new tasks
                del task_assignments[worker_id]
                
                # Send acknowledgment to worker
                await coordinator.send_message(
                    recipient=worker_id,
                    intent="task_acknowledgment",
                    payload={
                        "message": "Task completion acknowledged",
                        "status": "success"
                    }
                )
    
    # Run the coordinator
    try:
        logger.info("Coordinator is running and waiting for workers...")
        await coordinator.run()
    except Exception as e:
        logger.error(f"Coordinator error: {e}")
    finally:
        await coordinator.disconnect()

# Worker Agent
async def run_worker(worker_id, registry_url, capabilities):
    # Create a client for the worker
    worker = UAP_Client(entity_id=worker_id, registry_url=registry_url)
    
    # Connect to the registry
    await worker.connect()
    logger.info(f"Worker {worker_id} connected to registry")
    
    # Register message handlers
    @worker.message_handler(intent="task_assignment")
    async def handle_task_assignment(message):
        task_id = message.payload.get("task_id")
        task_name = message.payload.get("task_name")
        duration = message.payload.get("duration")
        
        logger.info(f"Worker {worker_id} received task: {task_name} (ID: {task_id})")
        
        # Simulate working on the task
        logger.info(f"Worker {worker_id} is working on {task_name}...")
        await asyncio.sleep(duration)
        
        # Generate a result
        result = {
            "summary": f"Completed {task_name} with {random.randint(85, 99)}% accuracy",
            "details": {
                "processing_time": duration,
                "worker_id": worker_id,
                "timestamp": time.time()
            }
        }
        
        # Report task completion
        await worker.send_message(
            recipient="coordinator",
            intent="task_completed",
            payload={
                "task_id": task_id,
                "result": result
            }
        )
        
        logger.info(f"Worker {worker_id} completed task: {task_name}")
    
    @worker.message_handler(intent="no_task_available")
    async def handle_no_task(message):
        retry_after = message.payload.get("retry_after", 5)
        logger.info(f"Worker {worker_id} has no task. Retrying after {retry_after} seconds...")
        await asyncio.sleep(retry_after)
        
        # Request a new task
        await worker.send_message(
            recipient="coordinator",
            intent="worker_ready",
            payload={"capabilities": capabilities}
        )
    
    @worker.message_handler(intent="task_acknowledgment")
    async def handle_acknowledgment(message):
        logger.info(f"Worker {worker_id} received acknowledgment: {message.payload.get('message')}")
        
        # Request a new task
        await worker.send_message(
            recipient="coordinator",
            intent="worker_ready",
            payload={"capabilities": capabilities}
        )
    
    @worker.message_handler(intent="collaboration_completed")
    async def handle_collaboration_completed(message):
        logger.info(f"Worker {worker_id} received collaboration completion notification")
        logger.info(f"Message: {message.payload.get('message')}")
        
        # Worker can disconnect or prepare for new collaborations
        logger.info(f"Worker {worker_id} is shutting down...")
        await worker.disconnect()
    
    # Announce readiness to the coordinator
    await worker.send_message(
        recipient="coordinator",
        intent="worker_ready",
        payload={"capabilities": capabilities}
    )
    
    # Run the worker
    try:
        logger.info(f"Worker {worker_id} is running...")
        await worker.run()
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")
    finally:
        await worker.disconnect()

# Monitor Agent
async def run_monitor(registry_url):
    # Create a client for the monitor
    monitor = UAP_Client(entity_id="monitor", registry_url=registry_url)
    
    # Connect to the registry
    await monitor.connect()
    logger.info("Monitor connected to registry")
    
    # Task tracking
    collaborations = {}
    
    # Register message handlers
    @monitor.message_handler(intent="task_update")
    async def handle_task_update(message):
        task_id = message.payload.get("task_id")
        status = message.payload.get("status")
        worker_id = message.payload.get("worker_id")
        timestamp = message.payload.get("timestamp")
        collaboration_id = message.payload.get("collaboration_id")
        
        # Initialize collaboration tracking if needed
        if collaboration_id not in collaborations:
            collaborations[collaboration_id] = {
                "tasks": {},
                "start_time": timestamp,
                "status": "in_progress"
            }
        
        # Update task status
        collaborations[collaboration_id]["tasks"][task_id] = {
            "status": status,
            "worker_id": worker_id,
            "last_update": timestamp
        }
        
        # Log the update
        logger.info(f"Monitor: Task {task_id} is now {status}, assigned to {worker_id}")
        
        # Calculate and log progress
        collab = collaborations[collaboration_id]
        total_tasks = len(TASKS)
        completed_tasks = sum(1 for t in collab["tasks"].values() if t["status"] == "completed")
        progress = (completed_tasks / total_tasks) * 100
        
        logger.info(f"Monitor: Collaboration {collaboration_id} progress: {progress:.1f}% ({completed_tasks}/{total_tasks} tasks)")
    
    @monitor.message_handler(intent="collaboration_completed")
    async def handle_collaboration_completed(message):
        collaboration_id = message.payload.get("collaboration_id")
        timestamp = message.payload.get("timestamp")
        results = message.payload.get("results", {})
        
        if collaboration_id in collaborations:
            # Update collaboration status
            collaborations[collaboration_id]["status"] = "completed"
            collaborations[collaboration_id]["end_time"] = timestamp
            collaborations[collaboration_id]["results"] = results
            
            # Calculate duration
            duration = timestamp - collaborations[collaboration_id]["start_time"]
            
            # Log completion
            logger.info(f"Monitor: Collaboration {collaboration_id} completed!")
            logger.info(f"Monitor: Duration: {duration:.2f} seconds")
            logger.info(f"Monitor: Results summary:")
            
            for task_id, result in results.items():
                if result and "summary" in result:
                    logger.info(f"  - {task_id}: {result['summary']}")
    
    # Run the monitor
    try:
        logger.info("Monitor is running and tracking collaborations...")
        await monitor.run()
    except Exception as e:
        logger.error(f"Monitor error: {e}")
    finally:
        await monitor.disconnect()

# Main function
async def main():
    # Registry URL
    registry_url = "localhost:8000"
    
    # Start the monitor
    monitor_task = asyncio.create_task(run_monitor(registry_url))
    
    # Start the coordinator
    coordinator_task = asyncio.create_task(run_coordinator(registry_url))
    
    # Give the coordinator and monitor time to initialize
    await asyncio.sleep(2)
    
    # Start the workers with different capabilities
    worker_tasks = [
        asyncio.create_task(run_worker("worker1", registry_url, {"skill_level": 8, "specialties": ["data", "analysis"]})),
        asyncio.create_task(run_worker("worker2", registry_url, {"skill_level": 6, "specialties": ["visualization", "reporting"]})),
        asyncio.create_task(run_worker("worker3", registry_url, {"skill_level": 4, "specialties": ["collection", "processing"]})),
    ]
    
    # Wait for all tasks to complete
    await asyncio.gather(coordinator_task, monitor_task, *worker_tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
