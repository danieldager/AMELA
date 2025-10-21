#!/usr/bin/env python3
"""
Analyze ASR pipeline results
Provides statistics and quality checks for transcriptions
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


def load_results(filepath):
    """Load ASR results from JSON file."""
    results = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}", file=sys.stderr)
    return results


def analyze_results(results):
    """Generate statistics from ASR results."""
    
    total_files = len(results)
    
    # Count files with transcriptions
    transcribed = [r for r in results if r.get('text', '').strip()]
    empty = [r for r in results if not r.get('text', '').strip()]
    
    # Text statistics
    text_lengths = [len(r.get('text', '')) for r in transcribed]
    word_counts = [len(r.get('text', '').split()) for r in transcribed]
    
    # Duration statistics
    durations = [r.get('duration', 0) for r in results]
    total_duration = sum(durations)
    
    # Calculate words per second
    wps_list = []
    for r in transcribed:
        duration = r.get('duration', 0)
        word_count = len(r.get('text', '').split())
        if duration > 0:
            wps_list.append(word_count / duration)
    
    stats = {
        'total_files': total_files,
        'transcribed': len(transcribed),
        'empty': len(empty),
        'success_rate': len(transcribed) / total_files * 100 if total_files > 0 else 0,
        'total_duration': total_duration,
        'avg_duration': sum(durations) / len(durations) if durations else 0,
        'total_chars': sum(text_lengths),
        'avg_chars': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'total_words': sum(word_counts),
        'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
        'avg_words_per_second': sum(wps_list) / len(wps_list) if wps_list else 0,
    }
    
    return stats, empty


def print_stats(stats):
    """Print statistics in a formatted way."""
    print("=" * 60)
    print("ASR Results Analysis")
    print("=" * 60)
    print(f"\nFile Statistics:")
    print(f"  Total files:           {stats['total_files']:,}")
    print(f"  Successfully transcribed: {stats['transcribed']:,}")
    print(f"  Empty transcriptions:  {stats['empty']:,}")
    print(f"  Success rate:          {stats['success_rate']:.2f}%")
    
    print(f"\nAudio Duration:")
    print(f"  Total duration:        {stats['total_duration']:.2f} seconds ({stats['total_duration']/3600:.2f} hours)")
    print(f"  Average duration:      {stats['avg_duration']:.2f} seconds")
    
    print(f"\nTranscription Statistics:")
    print(f"  Total characters:      {stats['total_chars']:,}")
    print(f"  Average chars/file:    {stats['avg_chars']:.2f}")
    print(f"  Total words:           {stats['total_words']:,}")
    print(f"  Average words/file:    {stats['avg_words']:.2f}")
    print(f"  Average words/second:  {stats['avg_words_per_second']:.2f}")
    print("=" * 60)


def print_samples(results, n=5):
    """Print sample transcriptions."""
    print(f"\nSample Transcriptions (first {n}):")
    print("-" * 60)
    for i, r in enumerate(results[:n], 1):
        filepath = Path(r['audio_filepath']).name
        text = r.get('text', '(empty)')
        duration = r.get('duration', 0)
        print(f"{i}. {filepath} ({duration:.2f}s)")
        print(f"   {text}")
        print()


def print_empty_files(empty_results):
    """Print files with empty transcriptions."""
    if empty_results:
        print(f"\nFiles with Empty Transcriptions ({len(empty_results)}):")
        print("-" * 60)
        for r in empty_results[:10]:  # Show first 10
            filepath = Path(r['audio_filepath']).name
            duration = r.get('duration', 0)
            print(f"  - {filepath} ({duration:.2f}s)")
        
        if len(empty_results) > 10:
            print(f"  ... and {len(empty_results) - 10} more")
    else:
        print("\n✓ All files successfully transcribed!")


def export_to_csv(results, output_path):
    """Export results to CSV format."""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['audio_filepath', 'duration', 'text', 'word_count', 'char_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results:
            text = r.get('text', '')
            writer.writerow({
                'audio_filepath': r.get('audio_filepath', ''),
                'duration': r.get('duration', 0),
                'text': text,
                'word_count': len(text.split()),
                'char_count': len(text)
            })
    
    print(f"\n✓ Exported to CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ASR pipeline results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to ASR results JSON file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample transcriptions to display (default: 5)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Show files with empty transcriptions"
    )
    
    args = parser.parse_args()
    
    # Load results
    try:
        results = load_results(args.results_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not results:
        print("Error: No results found in file", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    stats, empty = analyze_results(results)
    
    # Print statistics
    print_stats(stats)
    
    # Print samples
    if stats['transcribed'] > 0:
        print_samples([r for r in results if r.get('text', '').strip()], args.samples)
    
    # Print empty files if requested
    if args.show_empty:
        print_empty_files(empty)
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(results, args.export_csv)


if __name__ == "__main__":
    main()
