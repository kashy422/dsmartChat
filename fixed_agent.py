def sync_session_history(self, session_id):
    """Synchronize OpenAI messages with session history"""
    try:
        logger.info(f"SYNC: Starting history sync for session {session_id}")
        history = get_session_history(session_id)
        
        # Initialize messages for this session if not exists
        if session_id not in self.messages_by_session:
            self.messages_by_session[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        messages = self.messages_by_session[session_id]
        
        # Log current state
        logger.info(f"SYNC: Current OpenAI messages: {len(messages)}")
        logger.info(f"SYNC: Current history messages: {len(history.messages)}")
        logger.info("SYNC: OpenAI message types: " + ", ".join([f"{m['role']}" for m in messages]))
        logger.info("SYNC: History message types: " + ", ".join([f"{m['type']}" for m in history.messages]))
        
        # Count messages by type
        openai_counts = {"user": 0, "assistant": 0, "system": 0, "tool": 0}
        history_counts = {"human": 0, "ai": 0, "system": 0, "tool": 0}
        
        for msg in messages:
            msg_role = msg['role']
            if isinstance(msg.get('tool_calls'), list):
                openai_counts['tool'] += 1
            else:
                openai_counts[msg_role] += 1
                
        for msg in history.messages:
            history_counts[msg['type']] += 1
            
        logger.info(f"SYNC: OpenAI message counts: {openai_counts}")
        logger.info(f"SYNC: History message counts: {history_counts}")
        
        # Check for mismatch
        if len(messages) != len(history.messages) + 1:  # +1 for system message
            logger.warning(f"SYNC: Message count mismatch - OpenAI: {len(messages)}, History: {len(history.messages)}")
            
            # Log the actual messages for comparison
            logger.info("SYNC: OpenAI Messages:")
            for i, msg in enumerate(messages):
                content_preview = ""
                if msg.get('content'):
                    content_preview = msg['content'][:50] + "..."
                elif msg.get('tool_calls'):
                    tool_calls = msg['tool_calls']
                    if isinstance(tool_calls, list):
                        content_preview = f"Tool calls: {[t.function.name if hasattr(t, 'function') else t['function']['name'] for t in tool_calls]}"
                    else:
                        content_preview = "Tool calls present but not in expected format"
                logger.info(f"  {i}: {msg['role']} - {content_preview}")
                
            logger.info("SYNC: History Messages:")
            for i, msg in enumerate(history.messages):
                logger.info(f"  {i}: {msg['type']} - {msg.get('content', '')[:50]}...")
            
            # Rebuild messages from history
            new_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            for msg in history.messages:
                if msg['type'] == 'human':
                    new_messages.append({"role": "user", "content": msg['content']})
                elif msg['type'] == 'ai':
                    if msg.get('tool_calls'):
                        # Handle tool calls in the correct format
                        tool_calls = msg['tool_calls']
                        formatted_tool_calls = []
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                formatted_tool_calls.append({
                                    "id": tool_call.get('id', str(uuid.uuid4())),
                                    "type": "function",
                                    "function": {
                                        "name": tool_call['function']['name'],
                                        "arguments": tool_call['function']['arguments']
                                    }
                                })
                        new_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": formatted_tool_calls
                        })
                    else:
                        new_messages.append({"role": "assistant", "content": msg['content']})
                elif msg['type'] == 'tool':
                    new_messages.append({
                        "role": "tool",
                        "content": msg.get('content'),
                        "tool_call_id": msg.get('tool_call_id'),
                        "name": msg.get('name')
                    })
            
            # Update the session messages
            self.messages_by_session[session_id] = new_messages
            logger.info(f"SYNC: Rebuilt messages - new count: {len(new_messages)}")
            return self.messages_by_session[session_id]
        
        return messages
        
    except Exception as e:
        logger.error(f"SYNC ERROR: Failed to sync session history: {str(e)}", exc_info=True)
        # Return default messages if sync fails
        return [{"role": "system", "content": SYSTEM_PROMPT}] 