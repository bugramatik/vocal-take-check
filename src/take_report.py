import json
import os


def _format_seconds(value):
    minutes, seconds = divmod(float(value), 60.0)
    return f'{int(minutes)}:{seconds:05.2f}'


def summary_line(result):
    """One line a person can read in a terminal or under the generated cover."""
    flagged = result.get('flagged_ranges') or []
    if flagged:
        ranges = ', '.join(f'{_format_seconds(begin)}-{_format_seconds(end)}' for begin, end in flagged)
        return (f'Take score {result["score"]:.2f} | median drift {result["median_drift_cents"]:.0f} cents '
                f'| {len(flagged)} flagged range(s): {ranges}')
    return (f'Take score {result["score"]:.2f} | median drift {result["median_drift_cents"]:.0f} cents '
            f'| no flagged ranges')


def ranges_block(result):
    """The flagged ranges as a short text block for the WebUI textbox."""
    lines = [summary_line(result)]
    flagged = result.get('flagged_ranges') or []
    if not flagged:
        lines.append('Every voiced frame stayed within 60 cents of the source pitch.')
        return '\n'.join(lines)

    lines.append('')
    lines.append('Flagged ranges (start - end, seconds):')
    for begin, end in flagged:
        lines.append(f'  {_format_seconds(begin)} - {_format_seconds(end)}  ({end - begin:.2f} s)')
    return '\n'.join(lines)


def write_report(result, report_path, source_path=None, generated_path=None):
    """Write the take quality result next to the cover and return the report path."""
    payload = {
        'score': result['score'],
        'median_drift_cents': result['median_drift_cents'],
        'pitch_correlation': result['pitch_correlation'],
        'voiced_seconds': result['voiced_seconds'],
        'flagged_seconds': result['flagged_seconds'],
        'flagged_ranges': [{'start': begin, 'end': end} for begin, end in result.get('flagged_ranges') or []],
        'summary': summary_line(result),
    }
    if source_path:
        payload['source'] = os.path.basename(source_path)
    if generated_path:
        payload['generated'] = os.path.basename(generated_path)

    directory = os.path.dirname(report_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(report_path, 'w', encoding='utf8') as outfile:
        json.dump(payload, outfile, indent=2)

    return report_path
