import os
import json
import argparse
from tree_sitter import Language, Parser
from pathlib import Path
from treelib import Node, Tree
from tqdm import tqdm


language_list = ['bash', 'csharp', 'cpp', 'java', 'php', 'python', 'rust', 'swift', 'typescript']

function_node_name = {
    "cpp": ['function_definition'], # https://github.com/tree-sitter/tree-sitter-cpp/blob/master/grammar.js
    "csharp": ['method_declaration'], # https://github.com/tree-sitter/tree-sitter-c-sharp/blob/master/grammar.js
    "java": ['method_declaration'], # https://github.com/tree-sitter/tree-sitter-java/blob/master/grammar.js
    "php": ['method_declaration'], # https://github.com/tree-sitter/tree-sitter-php/blob/master/grammar.js
    "python": ['function_definition'], # https://github.com/tree-sitter/tree-sitter-python/blob/master/grammar.js
    "rust": ['function_item'], # https://github.com/tree-sitter/tree-sitter-rust/blob/master/grammar.js
    "swift": ['function_declaration'], # https://github.com/alex-pinkus/tree-sitter-swift/blob/main/grammar.js
    "typescript": ['function_declaration', 'method_definition'], # https://github.com/tree-sitter/tree-sitter-typescript/blob/master/typescript/grammar.js
    "bash": ['function_definition'] # https://github.com/tree-sitter/tree-sitter-bash/blob/master/grammar.js
}

comment_node_name = {
    "cpp": ['comment'], # https://github.com/tree-sitter/tree-sitter-cpp/blob/master/grammar.js
    "csharp": ['comment'], # https://github.com/tree-sitter/tree-sitter-c-sharp/blob/master/grammar.js
    "java": ['comment', 'block_comment', 'line_comment'], # https://github.com/tree-sitter/tree-sitter-java/blob/master/grammar.js
    "php": ['comment'], # https://github.com/tree-sitter/tree-sitter-php/blob/master/grammar.js
    "python": ['comment'], # https://github.com/tree-sitter/tree-sitter-python/blob/master/grammar.js
    "rust": ['line_comment', 'block_comment'], # https://github.com/tree-sitter/tree-sitter-rust/blob/master/grammar.js
    "swift": ['comment', 'multiline_comment'], # https://github.com/alex-pinkus/tree-sitter-swift/blob/main/grammar.js
    "typescript": ['comment'], # https://github.com/tree-sitter/tree-sitter-typescript/blob/master/typescript/grammar.js
    "bash": ['comment'] # https://github.com/tree-sitter/tree-sitter-bash/blob/master/grammar.js
}


def strip_c_style_comment_delimiters(comment: str) -> str:
    comment_lines = comment.split('\n')
    cleaned_lines = []
    for l in comment_lines:
        l = l.strip()
        if l.endswith('*/'):
            l = l[:-2]
        if l.startswith('*'):
            l = l[1:]
        elif l.startswith('/**'):
            l = l[3:]
        elif l.startswith('/*'):
            l = l[2:]
        elif l.startswith('///'):
            l = l[3:]
        elif l.startswith('//'):
            l = l[2:]
        cleaned_lines.append(l.strip())
    return '\n'.join(cleaned_lines)


def get_docstring_summary(docstring: str) -> str:
    """Get the first lines of the documentation comment up to the empty lines."""
    if '\n\n' in docstring:
        return docstring.split('\n\n')[0]
    elif '@' in docstring:
        return docstring[:docstring.find('@')]  # This usually is the start of a JavaDoc-style @param comment.
    return docstring


def extract_methods(node, code, methods, lang):
    if len(node.children) == 0:
        if node.type in function_node_name[lang]:
            methods.append({"content": code[node.start_byte : node.end_byte].decode('UTF-8'), "range": list(range(node.start_point[0]+1, node.end_point[0]+2)), "start_byte": node.start_byte, "end_byte": node.end_byte, "type": node.type, "node": node})
    for child in node.children:
        if child.type in function_node_name[lang]:
            methods.append({"content": code[child.start_byte : child.end_byte].decode('UTF-8'), "range": list(range(child.start_point[0]+1, child.end_point[0]+2)), "start_byte": child.start_byte, "end_byte": child.end_byte, "type": child.type, "node": child})
        methods = extract_methods(child, code, methods, lang)
    return methods


def extract_comments(node, code, comments, lang):
    if len(node.children) == 0:
        if node.type in comment_node_name[lang]:
            comment_dict = {"content": code[node.start_byte : node.end_byte].decode('UTF-8'), "range": list(range(node.start_point[0]+1, node.end_point[0]+2)), "start_byte": node.start_byte, "end_byte": node.end_byte, "type": node.type}
            if comment_dict not in comments:
                comments.append(comment_dict)
    for child in node.children:
        if child.type in comment_node_name[lang]:
            comment_dict = {"content": code[child.start_byte : child.end_byte].decode('UTF-8'), "range": list(range(child.start_point[0]+1, child.end_point[0]+2)), "start_byte": child.start_byte, "end_byte": child.end_byte, "type": child.type}
            if comment_dict not in comments:
                comments.append(comment_dict)
        comments = extract_comments(child, code, comments, lang)
    return comments


def main(args):
    parsers = {}
    lib_folder = Path(args.treesitter_path)
    for lang in language_list:
        lib_path = lib_folder.joinpath(lang+".so")
        language = Language(lib_path, lang.replace("csharp", "c_sharp"))
        parser = Parser()
        parser.set_language(language)
        parsers[lang] = parser

    with open(args.datafile_path, 'r') as f:
        dataset_raw = f.readlines()

    function_comment_pairs = []
    success = 0
    if args.use_cover == "True":
        use_cover = True
    else:
        use_cover = False
    for snippet in tqdm(dataset_raw):
        snippet = json.loads(snippet)
        LANG = snippet['lang']
        if snippet['lang'] == "shell":
            LANG = "bash"
        # if LANG == "csharp" and snippet['max_stars_repo_path'].endswith(".cshtml"):
        #     continue
        parser = parsers[LANG]
        if snippet['seed'].startswith("<reponame>") or snippet['seed'].startswith("<filename>"):
            # remove the first line
            snippet['seed'] = snippet['seed'][snippet['seed'].index("\n") + 1 :]
        blob = snippet['content']
        if isinstance(blob, str):
            code = bytes(blob, "utf8")
        blob_tree = parser.parse(code)

        # extract all methods from content
        methods = []
        methods = extract_methods(blob_tree.root_node, code, methods, LANG)

        # extract all comments from content
        comments = []
        comments = extract_comments(blob_tree.root_node, code, comments, LANG)

        if len(comments) != 0:
            # sort all comments based on line number
            comments_sorted = sorted(comments, key=lambda x: x['range'][-1])

            # combine comments that are next to each other
            comment_groups = [[0]]
            for comment_idx in range(1, len(comments_sorted)):
                comment_comment_interval = code[comments_sorted[comment_idx-1]['end_byte']:comments_sorted[comment_idx]['start_byte']].decode('UTF-8')
                comment_comment_interval_clean = comment_comment_interval.replace("\n", "").replace(" ", "").replace("\r", "").replace("\t", "")
                if comment_comment_interval_clean == "" and comment_comment_interval.count("\n") == 1:
                    comment_groups[-1].append(comment_idx)
                else:
                    comment_groups.append([comment_idx])
            comments = []
            for comment_group in comment_groups:
                comment_dict = {"content": "\n".join([comments_sorted[comment_idx]["content"] for comment_idx in comment_group])}
                comment_dict["start_byte"] = comments_sorted[comment_group[0]]["start_byte"]
                comment_dict["end_byte"] = comments_sorted[comment_group[-1]]["end_byte"]
                comment_dict["range"] = []
                for comment_idx in comment_group:
                    comment_dict["range"] += comments_sorted[comment_idx]["range"]
                comments.append(comment_dict)

        # extract range of the seed
        content_lines = snippet['content'].splitlines(keepends=True)
        seed_lines = snippet['seed'].splitlines(keepends=True)
        seed_start_line = -1
        seed_end_line = -1
        for i in range(len(content_lines)-len(seed_lines)+1):
            if seed_start_line != -1:
                continue
            if content_lines[i:i+len(seed_lines)] == seed_lines:
                seed_start_line = i+1
                seed_end_line = i+len(seed_lines)
        assert content_lines[seed_start_line-1:seed_end_line] == seed_lines
        seed_range = list(range(seed_start_line, seed_end_line+1))

        # extract covered methods
        methods_covered = []
        for method in methods:
            if not set(method['range']).isdisjoint(seed_range):
                methods_covered.append(method)
        
        # extract docstring for covered methods
        for method in methods:
            if use_cover:
                if method not in methods_covered:
                    continue
            else:
                if method in methods_covered:
                    continue
            method["docstring"] = ''
            if LANG == "python":
                for block_node in method["node"].children:
                    if block_node.type == 'block':
                        docstring_node = [node for node in block_node.children if
                                node.type == 'expression_statement' and node.children[0].type == 'string']
                        if len(docstring_node) > 0:
                            docstring_node_extracted = docstring_node[0]
                            docstring = code[docstring_node_extracted.start_byte : docstring_node_extracted.end_byte].decode('UTF-8')
                            method["docstring"] = get_docstring_summary(strip_c_style_comment_delimiters(docstring.strip().strip('"').strip("'"))).strip()
                            success += 1
            else:
                comments_above = [comment for comment in comments if comment['range'][-1] < method['range'][0]]
                if len(comments_above) == 0:
                    continue
                comments_above_close = comments_above[-1]
                comment_method_interval = code[comments_above_close['end_byte']:method['start_byte']].decode('UTF-8')
                comment_method_interval_clean = comment_method_interval.replace("\n", "").replace(" ", "").replace("\r", "").replace("\t", "")
                if comment_method_interval_clean == "":
                    method["docstring"] = get_docstring_summary(strip_c_style_comment_delimiters(comments_above_close['content'])).strip()
                    success += 1
        
        # save results
        snippet['function'] = []
        for method in methods:
            if use_cover:
                if method not in methods_covered:
                    continue
            else:
                if method in methods_covered:
                    continue
            snippet['function'].append({"function": method["content"], "docstring": method["docstring"]})
        function_comment_pairs.append(snippet)
    # with open(args.output_path, 'w') as outfile:
    #     for entry in function_comment_pairs:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')
    print("#Function-docstring pairs:", success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile_path", type=str, default="/scratch/data-raw.jsonl")
    parser.add_argument("--output_path", type=str, default="/scratch/data-raw-with-function-docstring.jsonl")
    parser.add_argument("--treesitter_path", type=str, default="util/")
    parser.add_argument("--use_cover", type=str, default="True")
    args = parser.parse_args()
    main(args)
