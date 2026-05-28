import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader
from pageindex.mineru_assets import export_assets_from_mineru

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')
    parser.add_argument('--if-add-medical-metadata', type=str, default=None,
                      help='Whether to extract medical article metadata (yes/no)')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()
    
    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Process PDF file
        user_opt = {
            'model': args.model,
            'toc_check_page_num': args.toc_check_pages,
            'max_page_num_each_node': args.max_pages_per_node,
            'max_token_num_each_node': args.max_tokens_per_node,
            'if_add_node_id': args.if_add_node_id,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_medical_metadata': args.if_add_medical_metadata,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        # Process the PDF
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results into folder: results/<pdf_name>/
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
        output_dir = './results'
        output_folder = os.path.join(output_dir, pdf_name)
        output_file = os.path.join(output_folder, 'structure.json')
        os.makedirs(output_folder, exist_ok=True)

        # Export figure/table assets through MinerU markdown pipeline
        try:
            project_root = os.path.abspath('.')
            markdown_output_dir = os.path.join('.', 'documents', 'markdown')
            figures_info, tables_info, mineru_md_path = export_assets_from_mineru(
                pdf_path=args.pdf_path,
                result_folder=output_folder,
                project_root=project_root,
                markdown_output_dir=markdown_output_dir,
                model=opt.model,
            )
            # Store figure/table metadata separately under assets/figures.json and assets/tables.json
            assets_dir = os.path.join(output_folder, 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            figures_file = os.path.join(assets_dir, 'figures.json')
            tables_file = os.path.join(assets_dir, 'tables.json')
            with open(figures_file, 'w', encoding='utf-8') as ff:
                json.dump({"figures_info": figures_info}, ff, indent=2, ensure_ascii=False)
            with open(tables_file, 'w', encoding='utf-8') as tf:
                json.dump({"tables_info": tables_info}, tf, indent=2, ensure_ascii=False)

            # Keep structure.json focused on article structure + paths only
            toc_with_page_number.pop('figures_info', None)
            toc_with_page_number.pop('tables_info', None)
            toc_with_page_number['raw_md_path'] = os.path.relpath(
                mineru_md_path, os.path.abspath('.')
            ).replace("\\", "/")
            toc_with_page_number['figures_info_path'] = os.path.relpath(
                figures_file, os.path.abspath('.')
            ).replace("\\", "/")
            toc_with_page_number['tables_info_path'] = os.path.relpath(
                tables_file, os.path.abspath('.')
            ).replace("\\", "/")
            toc_with_page_number['assets_info_path'] = os.path.relpath(
                figures_file, os.path.abspath('.')
            ).replace("\\", "/")
        except Exception as e:
            print(f'Warning: failed to export PDF assets: {e}')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id,
            'if_add_medical_metadata': args.if_add_medical_metadata,
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id,
            summary_language=getattr(opt, "summary_language", "auto"),
            if_add_medical_metadata=getattr(opt, "if_add_medical_metadata", "no"),
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]
        output_dir = './results'
        output_folder = os.path.join(output_dir, md_name)
        output_file = os.path.join(output_folder, 'structure.json')
        os.makedirs(output_folder, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')