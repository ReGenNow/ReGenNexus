"""
ReGenNexus Core - Access Control Policy Module

This module implements policy-based access control for ReGenNexus Core,
providing fine-grained permission management for entities.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)

class PolicyManager:
    """
    Manages access control policies for ReGenNexus Core.
    
    Provides policy definition, evaluation, and enforcement for
    controlling entity access to resources and operations.
    """
    
    def __init__(self):
        """Initialize the policy manager."""
        self.policies = {}
        self.entity_roles = {}
        self.role_permissions = {}
    
    async def add_policy(self, policy_id: str, policy_def: Dict[str, Any]):
        """
        Add a new policy.
        
        Args:
            policy_id: Unique identifier for the policy
            policy_def: Policy definition
        """
        self.policies[policy_id] = policy_def
        logger.info(f"Policy added: {policy_id}")
    
    async def remove_policy(self, policy_id: str):
        """
        Remove a policy.
        
        Args:
            policy_id: Identifier of the policy to remove
        """
        if policy_id in self.policies:
            del self.policies[policy_id]
            logger.info(f"Policy removed: {policy_id}")
    
    async def assign_role(self, entity_id: str, role: str):
        """
        Assign a role to an entity.
        
        Args:
            entity_id: Identifier of the entity
            role: Role to assign
        """
        if entity_id not in self.entity_roles:
            self.entity_roles[entity_id] = set()
        
        self.entity_roles[entity_id].add(role)
        logger.info(f"Role {role} assigned to entity {entity_id}")
    
    async def revoke_role(self, entity_id: str, role: str):
        """
        Revoke a role from an entity.
        
        Args:
            entity_id: Identifier of the entity
            role: Role to revoke
        """
        if entity_id in self.entity_roles and role in self.entity_roles[entity_id]:
            self.entity_roles[entity_id].remove(role)
            logger.info(f"Role {role} revoked from entity {entity_id}")
    
    async def define_role_permissions(self, role: str, permissions: List[str]):
        """
        Define permissions for a role.
        
        Args:
            role: Role to define permissions for
            permissions: List of permission strings
        """
        self.role_permissions[role] = set(permissions)
        logger.info(f"Permissions defined for role {role}: {permissions}")
    
    async def get_entity_permissions(self, entity_id: str) -> Set[str]:
        """
        Get all permissions for an entity.
        
        Args:
            entity_id: Identifier of the entity
            
        Returns:
            Set of permission strings
        """
        permissions = set()
        
        # Add permissions from roles
        if entity_id in self.entity_roles:
            for role in self.entity_roles[entity_id]:
                if role in self.role_permissions:
                    permissions.update(self.role_permissions[role])
        
        return permissions
    
    async def check_permission(self, entity_id: str, permission: str) -> bool:
        """
        Check if an entity has a specific permission.
        
        Args:
            entity_id: Identifier of the entity
            permission: Permission to check
            
        Returns:
            True if the entity has the permission, False otherwise
        """
        entity_permissions = await self.get_entity_permissions(entity_id)
        
        # Check for exact match
        if permission in entity_permissions:
            return True
        
        # Check for wildcard permissions
        for perm in entity_permissions:
            if self._match_wildcard_permission(perm, permission):
                return True
        
        return False
    
    def _match_wildcard_permission(self, pattern: str, permission: str) -> bool:
        """
        Check if a permission matches a wildcard pattern.
        
        Args:
            pattern: Permission pattern (may contain wildcards)
            permission: Permission to check
            
        Returns:
            True if the permission matches the pattern, False otherwise
        """
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"[^.]*").replace(":", r"\:")
        return bool(re.match(f"^{regex_pattern}$", permission))
    
    async def evaluate_policy(self, entity_id: str, resource: str, action: str,
                            context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Evaluate a policy for an entity.
        
        Args:
            entity_id: Identifier of the entity
            resource: Resource being accessed
            action: Action being performed
            context: Optional additional context
            
        Returns:
            True if access is allowed, False otherwise
        """
        # Check for direct permission
        permission = f"{resource}:{action}"
        if await self.check_permission(entity_id, permission):
            return True
        
        # Check policies
        ctx = context or {}
        for policy_id, policy in self.policies.items():
            if self._evaluate_policy_rules(policy, entity_id, resource, action, ctx):
                return True
        
        return False
    
    def _evaluate_policy_rules(self, policy: Dict[str, Any], entity_id: str,
                              resource: str, action: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate the rules in a policy.
        
        Args:
            policy: Policy definition
            entity_id: Identifier of the entity
            resource: Resource being accessed
            action: Action being performed
            context: Additional context
            
        Returns:
            True if access is allowed by the policy, False otherwise
        """
        # Check if the policy applies to this resource and action
        if "resources" in policy and resource not in policy["resources"]:
            return False
        
        if "actions" in policy and action not in policy["actions"]:
            return False
        
        # Check entity constraints
        if "entities" in policy:
            entities = policy["entities"]
            if isinstance(entities, list) and entity_id not in entities:
                return False
            elif isinstance(entities, dict):
                if "include" in entities and entity_id not in entities["include"]:
                    return False
                if "exclude" in entities and entity_id in entities["exclude"]:
                    return False
        
        # Check conditions
        if "conditions" in policy:
            for condition in policy["conditions"]:
                if not self._evaluate_condition(condition, entity_id, resource, action, context):
                    return False
        
        # If we got here, the policy allows access
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], entity_id: str,
                           resource: str, action: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition in a policy.
        
        Args:
            condition: Condition definition
            entity_id: Identifier of the entity
            resource: Resource being accessed
            action: Action being performed
            context: Additional context
            
        Returns:
            True if the condition is satisfied, False otherwise
        """
        condition_type = condition.get("type")
        
        if condition_type == "time_range":
            # Time-based condition
            import time
            current_time = context.get("current_time") or time.time()
            start_time = condition.get("start_time")
            end_time = condition.get("end_time")
            
            if start_time and current_time < start_time:
                return False
            if end_time and current_time > end_time:
                return False
            
            return True
            
        elif condition_type == "ip_range":
            # IP address condition
            client_ip = context.get("client_ip")
            if not client_ip:
                return False
            
            allowed_ips = condition.get("allowed_ips", [])
            for ip_range in allowed_ips:
                if self._ip_in_range(client_ip, ip_range):
                    return True
            
            return False
            
        elif condition_type == "attribute":
            # Entity attribute condition
            attribute = condition.get("attribute")
            value = condition.get("value")
            operator = condition.get("operator", "eq")
            
            entity_value = context.get("entity_attributes", {}).get(attribute)
            
            if operator == "eq":
                return entity_value == value
            elif operator == "ne":
                return entity_value != value
            elif operator == "gt":
                return entity_value > value
            elif operator == "lt":
                return entity_value < value
            elif operator == "in":
                return entity_value in value
            elif operator == "contains":
                return value in entity_value
            else:
                logger.warning(f"Unknown operator in condition: {operator}")
                return False
        
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False
    
    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """
        Check if an IP address is in a CIDR range.
        
        Args:
            ip: IP address to check
            ip_range: CIDR range
            
        Returns:
            True if the IP is in the range, False otherwise
        """
        # Simple implementation for demonstration
        # In a real implementation, use a proper IP address library
        if ip_range == ip:
            return True
        
        if "/" in ip_range:
            # This is a CIDR range
            base_ip, bits = ip_range.split("/")
            # Implement CIDR matching logic here
            # For now, just check if the base IP matches
            return ip.startswith(base_ip.rsplit(".", 1)[0])
        
        return False
