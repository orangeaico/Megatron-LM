"""
Loss mask processors for fine-tuning data preparation.

This module provides a modular system for determining which token indices
should have their loss mask set to 0 during training.
"""

from typing import List, Dict, Any, Protocol, Optional, Set
from abc import ABC, abstractmethod
import ast
import tokenize
import io

# Global debug flag
DEBUG_MODE = False

def set_debug_mode(enabled: bool):
    """Enable or disable debug logging."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


class LossMaskProcessor(ABC):
    """Abstract base class for loss mask processors."""
    
    @abstractmethod
    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any], 
                assistant_turn: int) -> List[int]:
        """
        Process the message and return indices that should be masked (loss=0).
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages (1-indexed)
            
        Returns:
            List of indices (relative to the message tokens) that should have loss mask set to 0
        """
        pass


class StrReplaceEditorProcessor(LossMaskProcessor):
    """
    Processor that checks for str_replace_editor actions and extracts fields.
    
    Parses XML content to determine command type and extract relevant fields.
    - For create commands: extracts file_text and identifies comment lines
    - For str_replace commands: extracts old_str and new_str
    - For view commands: returns empty list (ignore)
    """
    
    def _extract_parameter(self, content: str, param_name: str) -> Optional[str]:
        """Extract parameter value from XML content."""
        param_tag = f'<parameter={param_name}>'
        if param_tag not in content:
            return None
            
        param_start = content.find(param_tag) + len(param_tag)
        param_end = content.find('</parameter>', param_start)
        
        if param_end == -1:
            return None
            
        return content[param_start:param_end]
    
    def _strip_newlines(self, text: str) -> str:
        """Strip leading and trailing newlines from text."""
        if text.startswith('\n'):
            text = text[1:]
        if text.endswith('\n'):
            text = text[:-1]
        return text
    
    def _check_command_type(self, content: str, command: str) -> bool:
        """Check if content contains a command parameter with optional newlines."""
        # Check all possible variations with newlines
        variations = [
            f'<parameter=command>{command}</parameter>',
            f'<parameter=command>\n{command}</parameter>',
            f'<parameter=command>{command}\n</parameter>',
            f'<parameter=command>\n{command}\n</parameter>'
        ]
        return any(var in content for var in variations)
    
    def _get_comment_lines_from_ast(self, code: str) -> Set[int]:
        """Use AST to identify lines with comments (docstrings)."""
        comment_lines = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for docstrings in functions, classes, and modules
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Get the first statement in the body
                    if node.body and isinstance(node.body[0], ast.Expr):
                        value = node.body[0].value
                        if isinstance(value, (ast.Str, ast.Constant)) and isinstance(value.value if hasattr(value, 'value') else value.s, str):
                            # This is a docstring
                            comment_lines.add(node.body[0].lineno)
                            # Multi-line docstrings may span several lines
                            if hasattr(node.body[0], 'end_lineno') and node.body[0].end_lineno:
                                for line_num in range(node.body[0].lineno, node.body[0].end_lineno + 1):
                                    comment_lines.add(line_num)
                elif isinstance(node, ast.Module):
                    # Module-level docstring
                    if node.body and isinstance(node.body[0], ast.Expr):
                        value = node.body[0].value
                        if isinstance(value, (ast.Str, ast.Constant)) and isinstance(value.value if hasattr(value, 'value') else value.s, str):
                            comment_lines.add(node.body[0].lineno)
                            if hasattr(node.body[0], 'end_lineno') and node.body[0].end_lineno:
                                for line_num in range(node.body[0].lineno, node.body[0].end_lineno + 1):
                                    comment_lines.add(line_num)
        except:
            # If AST parsing fails, return empty set
            pass
        
        return comment_lines
    
    def _get_comment_lines_from_tokenizer(self, code: str) -> Set[int]:
        """Use tokenizer to identify lines with # comments (only pure comment lines, not inline comments)."""
        comment_lines = set()
        
        try:
            lines = code.split('\n')
            for line_num, line in enumerate(lines, 1):
                # Strip whitespace to check content
                stripped = line.lstrip()
                # Check if line starts with # (after stripping whitespace)
                if stripped.startswith('#'):
                    comment_lines.add(line_num)
        except:
            # If processing fails, return what we have
            pass
            
        return comment_lines
    
    def _get_all_comment_lines(self, code: str) -> Set[int]:
        """Get all comment lines including both # comments and docstrings."""
        # Get # comments using tokenizer
        hash_comments = self._get_comment_lines_from_tokenizer(code)
        # Get docstrings using AST
        docstrings = self._get_comment_lines_from_ast(code)
        # Combine both
        return hash_comments.union(docstrings)
    
    def _find_parameter_boundaries(self, tokenizer, token_ids: List[int], param_name: str, 
                                     window_size: int = 6) -> Optional[tuple[int, int]]:
        """
        Find the start and end positions of a parameter in the token sequence.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs to search in
            param_name: The parameter name to search for (e.g., 'file_text')
            window_size: Number of consecutive tokens to join for searching
            
        Returns:
            A tuple of (start_idx, end_idx) where:
            - start_idx is the position after <parameter=param_name>
            - end_idx is the position of </parameter>
            Returns None if parameter not found
        """
        # Convert all tokens to their string representation with special chars
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Clean tokens - convert special characters to actual representation
        cleaned_tokens = []
        for token in tokens:
            # Handle common tokenizer special characters
            clean = token.replace('Ġ', ' ')  # GPT-style space
            clean = clean.replace('▁', ' ')  # SentencePiece-style space
            clean = clean.replace('Ċ', '\n')  # Newline
            cleaned_tokens.append(clean)
        
        # Search for opening tag
        open_pattern = f'<parameter={param_name}>'
        start_idx = None
        
        # Find start position
        for i in range(len(cleaned_tokens) - window_size + 1):
            # Join consecutive tokens
            joined_text = ''.join(cleaned_tokens[i:i + window_size])
            
            # Check if search pattern is in the joined text
            if open_pattern in joined_text:
                # Find the exact position within the window
                prefix_len = joined_text.find(open_pattern) + len(open_pattern)
                
                # Estimate which token contains the end of the pattern
                cumulative_len = 0
                for j in range(window_size):
                    cumulative_len += len(cleaned_tokens[i + j])
                    if cumulative_len >= prefix_len:
                        # Found the position just after the parameter tag
                        start_idx = i + j + 1
                        break
                
                if start_idx is None:
                    # Fallback: use middle of window
                    start_idx = i + window_size // 2
                break
        
        if start_idx is None:
            return None
        
        # Search for closing tag
        close_pattern = '</parameter>'
        end_idx = None
        
        # Search from start_idx onwards
        for i in range(start_idx + 5, len(cleaned_tokens) - 4):
            # Join a few tokens to check for the pattern
            joined_text = ''.join(cleaned_tokens[i:i+4])
            if close_pattern in joined_text:
                # Found the closing tag
                end_idx = i
                break
        
        if end_idx is None:
            # If no closing tag found, return None (invalid parameter)
            return None
            
        return (start_idx, end_idx)
    
    def _find_token_indices_for_lines(self, tokenizer, token_ids: List[int], content: str, 
                                     target_lines: Set[int], file_text: str, offset: int = 0) -> List[int]:
        """Find token indices that correspond to specific lines in the file_text.
        
        Processes all comments (docstrings and single-line) in chronological order.
        
        Args:
            tokenizer: The tokenizer
            token_ids: List of token IDs to search within
            content: Not used in this version
            target_lines: Set of line numbers that contain comments
            file_text: The actual file text
            offset: Offset to add to returned indices (for when token_ids is a subset)
        
        Returns:
            List of token indices that should be masked
        """
        if not target_lines:
            return []
        
        file_lines = file_text.split('\n')
        masked_indices = []
        processed_lines = set()
        search_start_pos = 0
        
        # Process all comment lines in order
        sorted_lines = sorted(target_lines)
        
        # Group consecutive docstring lines together
        comment_groups = []
        i = 0
        while i < len(sorted_lines):
            line_num = sorted_lines[i]
            if line_num > len(file_lines):
                i += 1
                continue
                
            line_content = file_lines[line_num - 1]
            
            # Check if this is the start of a docstring
            if '"""' in line_content or "'''" in line_content:
                # Find the quote type
                quote_type = '"""' if '"""' in line_content else "'''"
                
                # Check if the docstring starts and ends on the same line
                quote_count = line_content.count(quote_type)
                if quote_count >= 2:
                    # Single-line docstring
                    comment_groups.append(('docstring', [line_num]))
                    i += 1
                else:
                    # Multi-line docstring - collect consecutive lines until closing quote
                    docstring_lines = [line_num]
                    j = i + 1
                    found_closing = False
                    
                    # Look for consecutive lines in target_lines that continue the docstring
                    while j < len(sorted_lines):
                        next_line_num = sorted_lines[j]
                        
                        # Check if this is the next consecutive line
                        expected_line = docstring_lines[-1] + 1
                        if next_line_num != expected_line:
                            # Not consecutive - this docstring is incomplete, treat first line as single comment
                            break
                        
                        if next_line_num <= len(file_lines):
                            next_content = file_lines[next_line_num - 1]
                            docstring_lines.append(next_line_num)
                            
                            # Check if this line closes the docstring
                            if quote_type in next_content:
                                found_closing = True
                                j += 1
                                break
                        j += 1
                    
                    if found_closing and len(docstring_lines) > 1:
                        comment_groups.append(('docstring', docstring_lines))
                        i = j
                    else:
                        # Treat as single line comment if no proper closing found
                        comment_groups.append(('single', [line_num]))
                        i += 1
            else:
                # Single line comment
                comment_groups.append(('single', [line_num]))
                i += 1
        
        if DEBUG_MODE:
            print(f"Processing {len(comment_groups)} comment groups in chronological order")
        
        # Process all comments in order
        for comment_type, lines in comment_groups:
            if comment_type == 'docstring':
                # Process docstring block
                first_line = lines[0] - 1  # 0-indexed
                last_line = lines[-1] - 1   # 0-indexed
                
                # Get the exact docstring text
                docstring_lines_text = file_lines[first_line:last_line + 1]
                docstring_text = '\n'.join(docstring_lines_text)
                
                if DEBUG_MODE:
                    print(f"Processing docstring block with {len(lines)} lines: {lines[:3]}...")
                
                # Tokenize the entire docstring
                docstring_tokens = tokenizer.encode(docstring_text, add_special_tokens=False)
                
                if not docstring_tokens:
                    continue
                
                # Try to find this token sequence
                found = False
                for start_offset in range(min(3, len(docstring_tokens))):
                    if found:
                        break
                    for end_offset in range(min(3, len(docstring_tokens))):
                        if found:
                            break
                        
                        search_tokens = docstring_tokens[start_offset:len(docstring_tokens)-end_offset if end_offset > 0 else None]
                        
                        if not search_tokens or len(search_tokens) < 3:
                            continue
                        
                        # Look for this sequence in token_ids
                        for i in range(search_start_pos, len(token_ids) - len(search_tokens) + 1):
                            if token_ids[i:i+len(search_tokens)] == search_tokens:
                                # Found match
                                found = True
                                start = i - start_offset
                                end = i + len(search_tokens) + end_offset
                                
                                # Ensure we don't go out of bounds
                                start = max(0, start)
                                end = min(len(token_ids), end)
                                
                                # Extend for whitespace as before
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                
                                # Extend backwards
                                for j in range(start - 1, max(0, start - 10), -1):
                                    if j < len(tokens):
                                        token_text = tokens[j]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        if '\n' in clean:
                                            break
                                        if clean.strip() == '':
                                            start = j
                                        else:
                                            break
                                
                                # Check if we need to extend forward
                                if end > 0 and end <= len(tokens):
                                    last_token = tokens[end - 1]
                                    last_clean = last_token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                    if '\n' not in last_clean:
                                        # Extend forwards for newline
                                        for j in range(end, min(len(tokens), end + 5)):
                                            if j < len(tokens):
                                                token_text = tokens[j]
                                                clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                if '\n' in clean and clean.strip() == '':
                                                    end = j + 1
                                                    break
                                                elif clean.strip() == '':
                                                    end = j + 1
                                                else:
                                                    break
                                
                                # Add tokens
                                for idx in range(start, end):
                                    masked_indices.append(idx + offset)
                                
                                search_start_pos = end
                                break
                
                if not found:
                    print(f"WARNING: Could not find docstring in token sequence! Lines: {lines}")
                    return []
            
            else:
                # Process single line comment
                line_num = lines[0]
                comment_line = file_lines[line_num - 1]
                
                if not comment_line and not comment_line.strip():
                    continue
                
                # Tokenize the comment line
                comment_tokens = tokenizer.encode(comment_line, add_special_tokens=False)
                
                if not comment_tokens:
                    continue
                
                if DEBUG_MODE and line_num == 1:
                    print(f"DEBUG: Processing line {line_num}: {repr(comment_line)}")
                    print(f"  Tokenized to {len(comment_tokens)} tokens: {comment_tokens}")
                    print(f"  Search starting at position {search_start_pos} in {len(token_ids)} total tokens")
                
                # Try to find this token sequence
                found = False
                for start_offset in range(min(3, len(comment_tokens))):
                    if found:
                        break
                    for end_offset in range(min(3, len(comment_tokens))):
                        if found:
                            break
                        
                        search_tokens = comment_tokens[start_offset:len(comment_tokens)-end_offset if end_offset > 0 else None]
                        
                        if not search_tokens:
                            continue
                        
                        # Look for this sequence in token_ids
                        for i in range(search_start_pos, len(token_ids) - len(search_tokens) + 1):
                            if token_ids[i:i+len(search_tokens)] == search_tokens:
                                # Found match
                                found = True
                                start = i - start_offset
                                end = i + len(search_tokens) + end_offset
                                
                                # Extend for whitespace as before
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                
                                # Extend backwards
                                for j in range(start - 1, max(0, start - 10), -1):
                                    if j < len(tokens):
                                        token_text = tokens[j]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        if '\n' in clean:
                                            break
                                        if clean.strip() == '':
                                            start = j
                                        else:
                                            break
                                
                                # Check if we need to extend forward
                                if end > 0 and end <= len(tokens):
                                    last_token = tokens[end - 1]
                                    last_clean = last_token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                    if '\n' not in last_clean:
                                        # Extend forwards for newline
                                        for j in range(end, min(len(tokens), end + 5)):
                                            if j < len(tokens):
                                                token_text = tokens[j]
                                                clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                if '\n' in clean and clean.strip() == '':
                                                    end = j + 1
                                                    break
                                                elif clean.strip() == '':
                                                    end = j + 1
                                                else:
                                                    break
                                
                                # Add tokens
                                for idx in range(start, end):
                                    masked_indices.append(idx + offset)
                                
                                search_start_pos = end
                                break
                
                if not found:
                    print(f"WARNING: Could not find line {line_num} in token sequence: {repr(comment_line)}")
                    return []
        
        # Remove duplicates and sort
        masked_indices = sorted(set(masked_indices))
        return masked_indices
    
    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any], 
                assistant_turn: int) -> List[int]:
        """
        Parse str_replace_editor actions and extract fields.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages
            
        Returns:
            Empty list if str_replace_editor view command is found,
            None to continue processing for other cases
        """
        content = msg_dict.get('content', '')
        
        # Check if content contains str_replace_editor function call
        if '<function=str_replace_editor>' not in content:
            return None
            
        # Determine the command type
        is_create = False
        is_str_replace = False
        is_view = False
        
        # Check for command parameter with variations
        if self._check_command_type(content, 'create'):
            is_create = True
        elif self._check_command_type(content, 'str_replace'):
            is_str_replace = True
        elif self._check_command_type(content, 'view'):
            is_view = True
        else:
            # Fallback: check if it's str_replace by looking for old_str parameter
            if '<parameter=old_str>' in content:
                is_str_replace = True
            # Check if it's create by looking for file_text parameter
            elif '<parameter=file_text>' in content:
                is_create = True
        
        # Handle view command - return empty list to ignore
        if is_view:
            return []
        
        # Handle create command
        if is_create:
            file_text = self._extract_parameter(content, 'file_text')
            if file_text is not None:
                # Strip newlines for processing
                file_text_stripped = self._strip_newlines(file_text)
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: Found create command (assistant turn {assistant_turn})")
                
                # Get all comment lines
                comment_lines = self._get_all_comment_lines(file_text_stripped)
                
                
                if comment_lines:
                    if DEBUG_MODE:
                        print(f"Found {len(comment_lines)} comment lines")
                        # Print the actual comment lines
                        file_lines = file_text_stripped.split('\n')
                        print(f"Comment lines being masked:")
                        for line_num in sorted(comment_lines):
                            if 1 <= line_num <= len(file_lines):
                                comment_content = file_lines[line_num - 1]  # Don't strip - keep whitespace
                                # Only print if line has content (even if it's just whitespace)
                                if comment_content or comment_content.strip():
                                    print(f"  Line {line_num}: {repr(comment_content)}")
                    
                    # First, we need to find where file_text appears in the token sequence
                    # Use the new robust method to find parameter boundaries
                    boundaries = self._find_parameter_boundaries(tokenizer, token_ids, 'file_text')
                    
                    if boundaries is None:
                        # Always print warnings, regardless of debug mode
                        print(f"WARNING: Could not find create command file_text content in token sequence")
                        return []
                    
                    file_text_start_idx, file_text_end_idx = boundaries
                    
                    if DEBUG_MODE:
                        print(f"Found file_text boundaries: start={file_text_start_idx}, end={file_text_end_idx}")
                    
                    # Extract only the tokens that are part of file_text
                    file_text_token_ids = token_ids[file_text_start_idx:file_text_end_idx]
                    
                    # Find token indices corresponding to comment lines within file_text
                    masked_indices = self._find_token_indices_for_lines(
                        tokenizer, file_text_token_ids, content, comment_lines, file_text_stripped, 
                        offset=file_text_start_idx
                    )
                    
                    # Always calculate expected character count for validation
                    total_comment_chars = 0
                    file_lines = file_text_stripped.split('\n')
                    for line_num in sorted(comment_lines):
                        if 1 <= line_num <= len(file_lines):
                            # Include the line content plus newline
                            total_comment_chars += len(file_lines[line_num - 1]) + 1  # +1 for newline
                    
                    if masked_indices:
                        
                        # Calculate total characters in masked tokens
                        tokens = tokenizer.convert_ids_to_tokens(token_ids)
                        total_masked_chars = 0
                        for idx in masked_indices:
                            if 0 <= idx < len(tokens):
                                token_text = tokens[idx]
                                clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                total_masked_chars += len(clean)
                        
                        if DEBUG_MODE:
                            print(f"Masking {len(masked_indices)} tokens for comment lines")
                            # Print the actual tokens being masked
                            print(f"Tokens being masked:")
                            
                            # Group consecutive tokens
                            if masked_indices:
                                consecutive_groups = []
                                current_group = [masked_indices[0]]
                                
                                for i in range(1, len(masked_indices)):
                                    if masked_indices[i] == masked_indices[i-1] + 1:
                                        # Consecutive token
                                        current_group.append(masked_indices[i])
                                    else:
                                        # New group
                                        consecutive_groups.append(current_group)
                                        current_group = [masked_indices[i]]
                                
                                # Add the last group
                                consecutive_groups.append(current_group)
                                
                                # Print each group
                                for group in consecutive_groups:
                                    # Build the text for this group
                                    group_text = ""
                                    for idx in group:
                                        if 0 <= idx < len(tokens):
                                            token_text = tokens[idx]
                                            # Convert to readable format
                                            clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                            group_text += clean
                                    
                                    # Print with token range
                                    if len(group) == 1:
                                        print(f"  Token {group[0]}: {repr(group_text)}")
                                    else:
                                        print(f"  Tokens {group[0]}-{group[-1]}: {repr(group_text)}")
                                
                                # Show character count comparison in debug mode
                                print(f"\nCharacter count comparison:")
                                print(f"  Total characters in identified comments: {total_comment_chars}")
                                print(f"  Total characters in masked tokens: {total_masked_chars}")
                                
                                if total_comment_chars > 0:
                                    diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                    threshold = 5
                                    print(f"  Difference: {diff_percent:.1f}%")
                                    if diff_percent > threshold:
                                        print(f"  WARNING: Exceeds {threshold}% threshold - will return empty list")
                        
                        # Check for character count mismatch (always check, not just in debug) - AFTER printing debug info
                        if total_comment_chars > 0:
                            diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                            threshold = 5
                            if diff_percent > threshold:
                                print(f"WARNING: Character count mismatch in create command! Masked tokens differ from identified comments by {diff_percent:.1f}%")
                                print(f"  Comment characters: {total_comment_chars}, Masked characters: {total_masked_chars}")
                                print(f"  Returning empty list due to excessive mismatch (>{threshold}%)")
                                return []
                        
                        return masked_indices
                    else:
                        # No tokens were masked but we had comments - this is a problem
                        if comment_lines and total_comment_chars > 0:
                            print(f"WARNING: Found {len(comment_lines)} comment lines ({total_comment_chars} chars) in create command but could not mask any tokens!")
                            if DEBUG_MODE:
                                print(f"Comment lines that were not masked:")
                                for line_num in sorted(comment_lines):
                                    if 1 <= line_num <= len(file_lines):
                                        print(f"  Line {line_num}: {repr(file_lines[line_num - 1])}")
                            return []
                        elif DEBUG_MODE:
                            print(f"Could not map comment lines to token indices")
                else:
                    if DEBUG_MODE:
                        print(f"No comment lines found in file")
                
                # Return empty list if no comments found or couldn't map them
                return []
            else:
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: Create command found but no file_text parameter (assistant turn {assistant_turn})")
                return []
        
        # Handle str_replace command
        elif is_str_replace:
            old_str = self._extract_parameter(content, 'old_str')
            new_str = self._extract_parameter(content, 'new_str')
            
            if old_str is not None or new_str is not None:
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: Found str_replace command (assistant turn {assistant_turn})")
                    if old_str is not None:
                        old_str_stripped = self._strip_newlines(old_str)
                        print("Full old_str content:")
                        print(old_str_stripped)
                        print("=" * 80)
                    if new_str is not None:
                        new_str_stripped = self._strip_newlines(new_str)
                        print("Full new_str content:")
                        print(new_str_stripped)
                        print("=" * 80)
                
                all_masked_indices = []
                
                # Process old_str if present
                if old_str is not None:
                    old_str_stripped = self._strip_newlines(old_str)
                    comment_lines = self._get_all_comment_lines(old_str_stripped)
                    
                    if comment_lines:
                        if DEBUG_MODE:
                            print(f"Found {len(comment_lines)} comment lines in old_str")
                            # Print the actual comment lines
                            file_lines = old_str_stripped.split('\n')
                            print(f"Comment lines being masked in old_str:")
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    comment_content = file_lines[line_num - 1]
                                    if comment_content or comment_content.strip():
                                        print(f"  Line {line_num}: {repr(comment_content)}")
                        
                        # Find old_str boundaries using the same method as file_text
                        boundaries = self._find_parameter_boundaries(tokenizer, token_ids, 'old_str')
                        
                        if boundaries is None:
                            print(f"WARNING: Could not find str_replace old_str content in token sequence")
                            return []
                        else:
                            old_str_start_idx, old_str_end_idx = boundaries
                            
                            if DEBUG_MODE:
                                print(f"Found old_str boundaries: start={old_str_start_idx}, end={old_str_end_idx}")
                            
                            # Extract only the tokens that are part of old_str
                            old_str_token_ids = token_ids[old_str_start_idx:old_str_end_idx]
                            
                            # Find token indices corresponding to comment lines within old_str
                            masked_indices = self._find_token_indices_for_lines(
                                tokenizer, old_str_token_ids, content, comment_lines, old_str_stripped, 
                                offset=old_str_start_idx
                            )
                            
                            # Calculate character counts for validation
                            total_comment_chars = 0
                            file_lines = old_str_stripped.split('\n')
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    total_comment_chars += len(file_lines[line_num - 1]) + 1
                            
                            if masked_indices:
                                # Calculate total characters in masked tokens
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                total_masked_chars = 0
                                for idx in masked_indices:
                                    if 0 <= idx < len(tokens):
                                        token_text = tokens[idx]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        total_masked_chars += len(clean)
                                
                                if DEBUG_MODE:
                                    print(f"Masking {len(masked_indices)} tokens for comment lines in old_str")
                                    # Print the actual tokens being masked
                                    print(f"Tokens being masked in old_str:")
                                    
                                    # Group consecutive tokens
                                    if masked_indices:
                                        consecutive_groups = []
                                        current_group = [masked_indices[0]]
                                        
                                        for i in range(1, len(masked_indices)):
                                            if masked_indices[i] == masked_indices[i-1] + 1:
                                                # Consecutive token
                                                current_group.append(masked_indices[i])
                                            else:
                                                # New group
                                                consecutive_groups.append(current_group)
                                                current_group = [masked_indices[i]]
                                        
                                        # Add the last group
                                        consecutive_groups.append(current_group)
                                        
                                        # Print each group
                                        for group in consecutive_groups:
                                            # Build the text for this group
                                            group_text = ""
                                            for idx in group:
                                                if 0 <= idx < len(tokens):
                                                    token_text = tokens[idx]
                                                    # Convert to readable format
                                                    clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                    group_text += clean
                                            
                                            # Print with token range
                                            if len(group) == 1:
                                                print(f"  Token {group[0]}: {repr(group_text)}")
                                            else:
                                                print(f"  Tokens {group[0]}-{group[-1]}: {repr(group_text)}")
                                        
                                        # Show character count comparison in debug mode
                                        print(f"\nCharacter count comparison for old_str:")
                                        print(f"  Total characters in identified comments: {total_comment_chars}")
                                        print(f"  Total characters in masked tokens: {total_masked_chars}")
                                        
                                        if total_comment_chars > 0:
                                            diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                            threshold = 5
                                            print(f"  Difference: {diff_percent:.1f}%")
                                            if diff_percent > threshold:
                                                print(f"  WARNING: Exceeds {threshold}% threshold - will return empty list")
                                
                                # Check for character count mismatch (always check, not just in debug) - AFTER printing debug info
                                if total_comment_chars > 0:
                                    diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                    threshold = 5
                                    if diff_percent > threshold:
                                        print(f"WARNING: Character count mismatch in str_replace old_str! Masked tokens differ from identified comments by {diff_percent:.1f}%")
                                        print(f"  Comment characters: {total_comment_chars}, Masked characters: {total_masked_chars}")
                                        print(f"  Returning empty list due to excessive mismatch (>{threshold}%)")
                                        return []
                                
                                all_masked_indices.extend(masked_indices)
                            else:
                                # No tokens were masked but we had comments - this is a problem
                                if comment_lines and total_comment_chars > 0:
                                    print(f"WARNING: Found {len(comment_lines)} comment lines ({total_comment_chars} chars) in str_replace old_str but could not mask any tokens!")
                                    if DEBUG_MODE:
                                        print(f"Comment lines in old_str that were not masked:")
                                        for line_num in sorted(comment_lines):
                                            if 1 <= line_num <= len(file_lines):
                                                print(f"  Line {line_num}: {repr(file_lines[line_num - 1])}")
                                    return []
                
                # Process new_str if present
                if new_str is not None:
                    new_str_stripped = self._strip_newlines(new_str)
                    comment_lines = self._get_all_comment_lines(new_str_stripped)
                    
                    if comment_lines:
                        if DEBUG_MODE:
                            print(f"Found {len(comment_lines)} comment lines in new_str")
                            # Print the actual comment lines
                            file_lines = new_str_stripped.split('\n')
                            print(f"Comment lines being masked in new_str:")
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    comment_content = file_lines[line_num - 1]
                                    if comment_content or comment_content.strip():
                                        print(f"  Line {line_num}: {repr(comment_content)}")
                        
                        # Find new_str boundaries using the same method as file_text
                        boundaries = self._find_parameter_boundaries(tokenizer, token_ids, 'new_str')
                        
                        if boundaries is None:
                            print(f"WARNING: Could not find str_replace new_str content in token sequence")
                            return []
                        else:
                            new_str_start_idx, new_str_end_idx = boundaries
                            
                            if DEBUG_MODE:
                                print(f"Found new_str boundaries: start={new_str_start_idx}, end={new_str_end_idx}")
                            
                            # Extract only the tokens that are part of new_str
                            new_str_token_ids = token_ids[new_str_start_idx:new_str_end_idx]
                            
                            # Find token indices corresponding to comment lines within new_str
                            masked_indices = self._find_token_indices_for_lines(
                                tokenizer, new_str_token_ids, content, comment_lines, new_str_stripped, 
                                offset=new_str_start_idx
                            )
                            
                            # Calculate character counts for validation
                            total_comment_chars = 0
                            file_lines = new_str_stripped.split('\n')
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    total_comment_chars += len(file_lines[line_num - 1]) + 1
                            
                            if masked_indices:
                                # Calculate total characters in masked tokens
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                total_masked_chars = 0
                                for idx in masked_indices:
                                    if 0 <= idx < len(tokens):
                                        token_text = tokens[idx]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        total_masked_chars += len(clean)
                                
                                if DEBUG_MODE:
                                    print(f"Masking {len(masked_indices)} tokens for comment lines in new_str")
                                    # Print the actual tokens being masked
                                    print(f"Tokens being masked in new_str:")
                                    
                                    # Group consecutive tokens
                                    if masked_indices:
                                        consecutive_groups = []
                                        current_group = [masked_indices[0]]
                                        
                                        for i in range(1, len(masked_indices)):
                                            if masked_indices[i] == masked_indices[i-1] + 1:
                                                # Consecutive token
                                                current_group.append(masked_indices[i])
                                            else:
                                                # New group
                                                consecutive_groups.append(current_group)
                                                current_group = [masked_indices[i]]
                                        
                                        # Add the last group
                                        consecutive_groups.append(current_group)
                                        
                                        # Print each group
                                        for group in consecutive_groups:
                                            # Build the text for this group
                                            group_text = ""
                                            for idx in group:
                                                if 0 <= idx < len(tokens):
                                                    token_text = tokens[idx]
                                                    # Convert to readable format
                                                    clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                    group_text += clean
                                            
                                            # Print with token range
                                            if len(group) == 1:
                                                print(f"  Token {group[0]}: {repr(group_text)}")
                                            else:
                                                print(f"  Tokens {group[0]}-{group[-1]}: {repr(group_text)}")
                                        
                                        # Show character count comparison in debug mode
                                        print(f"\nCharacter count comparison for new_str:")
                                        print(f"  Total characters in identified comments: {total_comment_chars}")
                                        print(f"  Total characters in masked tokens: {total_masked_chars}")
                                        
                                        if total_comment_chars > 0:
                                            diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                            threshold = 5
                                            print(f"  Difference: {diff_percent:.1f}%")
                                            if diff_percent > threshold:
                                                print(f"  WARNING: Exceeds {threshold}% threshold - will return empty list")
                                
                                # Check for character count mismatch (always check, not just in debug) - AFTER printing debug info
                                if total_comment_chars > 0:
                                    diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                    threshold = 5
                                    if diff_percent > threshold:
                                        print(f"WARNING: Character count mismatch in str_replace new_str! Masked tokens differ from identified comments by {diff_percent:.1f}%")
                                        print(f"  Comment characters: {total_comment_chars}, Masked characters: {total_masked_chars}")
                                        print(f"  Returning empty list due to excessive mismatch (>{threshold}%)")
                                        return []
                                
                                all_masked_indices.extend(masked_indices)
                            else:
                                # No tokens were masked but we had comments - this is a problem
                                if comment_lines and total_comment_chars > 0:
                                    print(f"WARNING: Found {len(comment_lines)} comment lines ({total_comment_chars} chars) in str_replace new_str but could not mask any tokens!")
                                    if DEBUG_MODE:
                                        print(f"Comment lines in new_str that were not masked:")
                                        for line_num in sorted(comment_lines):
                                            if 1 <= line_num <= len(file_lines):
                                                print(f"  Line {line_num}: {repr(file_lines[line_num - 1])}")
                                    return []
                
                # Remove duplicates and sort
                all_masked_indices = sorted(set(all_masked_indices))
                
                if DEBUG_MODE and all_masked_indices:
                    print(f"Total masked indices for str_replace: {len(all_masked_indices)}")
                
                return all_masked_indices
            else:
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: str_replace command found but no old_str or new_str parameters (assistant turn {assistant_turn})")
                return []
        
        # Return None to continue processing
        return None


class CommandGtProcessor(LossMaskProcessor):
    """
    Original processor that masks '>' tokens when preceded by 'command'.
    
    This is the existing logic from get_zero_loss_mask_indices.
    """
    
    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any], 
                assistant_turn: int) -> List[int]:
        """
        Find '>' tokens preceded by 'command' and mark them for masking.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages
            
        Returns:
            List of indices to mask, or None if not applicable
        """
        if assistant_turn != 1:
            return []
            
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_to_mask = '>'
        masked_ids = []
        
        for i, token in enumerate(tokens[:-1]):
            if token_to_mask in token and 'command' in ''.join(tokens[i-3:i+1]):
                masked_ids.append(i)
        
        return masked_ids


class LossMaskProcessorManager:
    """Manager class that handles multiple loss mask processors."""
    
    def __init__(self):
        """Initialize the processor manager with default processors."""
        self.processors: List[LossMaskProcessor] = []
        self._initialize_default_processors()
    
    def _initialize_default_processors(self):
        """Initialize the default set of processors."""
        # Add processors in order of priority
        self.processors.append(StrReplaceEditorProcessor())
        # Comment out CommandGtProcessor for now since the original logic is commented out
        # self.processors.append(CommandGtProcessor())
    
    def add_processor(self, processor: LossMaskProcessor):
        """Add a new processor to the manager."""
        self.processors.append(processor)
    
    def get_zero_loss_mask_indices(self, tokenizer, token_ids: List[int], 
                                   msg_dict: Dict[str, Any], assistant_turn: int) -> List[int]:
        """
        Process the message through all processors to get indices to mask.
        
        This is the main entry point that replaces the original get_zero_loss_mask_indices function.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages
            
        Returns:
            List of indices (relative to the message tokens) that should have loss mask set to 0
        """
        for processor in self.processors:
            result = processor.process(tokenizer, token_ids, msg_dict, assistant_turn)
            
            # If a processor returns a list (including empty list), use that result
            # If it returns None, continue to the next processor
            if result is not None:
                return result
        
        # If no processor handled the message, return empty list
        return []


# Global instance for easy access
loss_mask_processor_manager = LossMaskProcessorManager()


def get_zero_loss_mask_indices(tokenizer, token_ids: List[int], 
                               msg_dict: Dict[str, Any], assistant_turn: int) -> List[int]:
    """
    Main function to get indices that should have loss mask set to 0.
    
    This function serves as the single entry point from create_weighted_sft_dataset.py
    and internally handles all processors.
    
    Args:
        tokenizer: The tokenizer instance
        token_ids: List of token IDs for the message
        msg_dict: Dictionary containing 'role' and 'content' fields
        assistant_turn: The turn number for assistant messages
        
    Returns:
        List of indices (relative to the message tokens) that should have loss mask set to 0
    """
    return loss_mask_processor_manager.get_zero_loss_mask_indices(
        tokenizer, token_ids, msg_dict, assistant_turn
    )