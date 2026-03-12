===============================
Troubleshooting Guide
===============================

This guide covers common issues and best practices when using the RDM Toolkit for rail network analysis.

File Naming Issues with Downloaded Data
========================================

One of the most common issues arises from inconsistencies in file naming from the Rail Data Marketplace (RDM). These inconsistencies can stem from form updates, data corrections, or human errors during file uploads.

**Transparency File Name Variations**

The delay attribution data files are expected to follow the naming convention:

    ``Transparency_YY-YY_PXX.csv``

However, you may encounter files with spelling variations. For example:

- ``Transparency_23-24_P12.csv`` ✓ (correct)
- ``Transparancy_23-24_P12.csv`` ✗ (misspelled)

The most common variation is **"Transparancy"** where the vowels "a" and "e" are transposed. This can cause the preprocessor to fail silently if file detection is case-sensitive or pattern-based.

**What to Do**

- Before running the preprocessor, manually verify the spelling of all downloaded files in your ``demo/data/`` folder
- Rename any files with spelling errors to match the expected ``Transparency_YY-YY_PXX.csv`` pattern
- Check the RDM platform documentation for the correct file names in your downloaded archives
- If using automated scripts, validate file names programmatically before processing

**Additional Naming Considerations**

- Pay attention to the year-financial year format (e.g., "23-24" for April 2023 to March 2024)
- Verify the period notation (e.g., "P01" for April, "P12" for March)
- Note that some downloads may include date suffixes (e.g., "202324 data files 20250213.zip") where the date indicates the last data entry

Partial Station Processing
===========================

The preprocessor in this toolkit can process all stations or specific subsets based on category. **This is a deliberate feature** that allows users to test workflows or process data incrementally. However, incomplete processing has important implications for analysis.

**Impact of Partial Station Processing**

When you process only a subset of stations (e.g., Category A only, or specific individual stations), the following occurs:

1. **Limited Data Coverage**: Only the processed stations' data will be available in the ``processed_data/`` folder

2. **Altered Network Representation**: The demos will show a fragmented view of the network, not representative of the full system. Stations that were not processed will not appear in aggregated analyses

3. **Demo Limitations**: 
   - ✓ Station View demo will work (analyzes individual stations)
   - ✗ Aggregate View demo will be incomplete or show partial network statistics
   - ✗ Incident View demo will miss incidents at non-processed stations
   - ✗ Time View demo will show incomplete temporal patterns
   - ✗ Train View demo may have incomplete journey data

4. **Missing Error Outputs**: Some analyses or visualizations may fail with errors because expected data is unavailable. For example, network-wide statistics or delay propagation analysis cannot be computed without comprehensive station coverage

**Best Practice**

- Run the full preprocessor with ``--all-categories`` for comprehensive analysis:

    ``python -m rdmpy.preprocessor --all-categories``

- Only process partial datasets if you are:
  - Testing the workflow on a small subset
  - Conducting station-specific analysis
  - Operating under severe computational constraints

- **Document Your Processing Choice**: If you do process partial data, note which stations or categories were included. This prevents misinterpretation of results

- **Expect Incomplete Results**: Be aware that aggregate metrics and network-wide visualizations will not reflect the true state of the full network

Processing Time Considerations
===============================

**Be Aware of Execution Time**

- As of November 2025, processing all stations takes approximately **1 full day (24 hours)** to complete
- Processing is computationally intensive due to the volume of train schedule and delay data
- Do not interrupt the process unless necessary, as partial runs may require cleanup

**Recommendations**

- Run the full preprocessor during off-peak hours or overnight
- Monitor disk space: the processed data files can be substantial
- Run on a machine with adequate RAM (at least 8GB recommended) to avoid slowdowns
- Consider running on a server or high-performance machine if available

Data Validation Best Practices
==============================

**Before Running the Preprocessor**

1. **Verify All Required Files Are Present**
   - Ensure you have downloaded all months of delay data for your desired period
   - Check that the schedule file (``CIF_ALL_FULL_DAILY_toc-full_p4.pkl``) exists (after running cleaning)
   - Confirm the schedule cleaning step was completed: ``python data/schedule_cleaning.py``

2. **Check File Integrity**
   - Confirm that extracted .zip files contain the expected number of CSV files
   - Verify file sizes are reasonable (not corrupted or partial downloads)
   - Spot-check a few rows in the delay files to ensure proper formatting

3. **Validate File Spellings and Naming**
   - Use the checklist from the "File Naming Issues" section above
   - Create a small test with a single station first: ``python -m rdmpy.preprocessor <STANOX_CODE>``

**After Running the Preprocessor**

1. **Check Output**
   - Verify that files have been created in ``processed_data/``
   - Spot-check the demo notebooks to ensure data loads correctly
   - Look for any warning messages in the preprocessor logs

2. **Validate Data Completeness**
   - Check that the number of processed stations matches your expectations
   - Review the date ranges covered in processed files
   - Ensure no stations have all-zero or missing data

Common Error Messages and Solutions
====================================

**ModuleNotFoundError: No module named 'rdmpy'**

- Ensure you are running the preprocessor from the repository root directory
- Verify that the Python environment has the required packages installed (see Requirements section)
- Install missing dependencies: ``pip install -r requirements.txt``

**FileNotFoundError: Cannot find data files**

- Check that the ``demo/data/`` folder exists and contains the downloaded files
- Verify file names match the expected format (see File Naming Issues section)
- Ensure the schedule file has been cleaned and saved as ``CIF_ALL_FULL_DAILY_toc-full_p4.pkl``

**No OutputError or Empty Results in Demos**

- Check that the preprocessor completed successfully for all required stations
- Verify that ``processed_data/`` contains parquet files
- If processing partial stations, use the Station View demo which handles single-station data best

**Memory or Performance Issues During Preprocessing**

- Close other applications to free up RAM
- Consider processing by category instead of all stations at once
- Run on a machine with more available memory
- Check available disk space before starting

Network Analysis Considerations
================================

**When Interpreting Results**

1. **Account for Data Lag**: The delay attribution data from RDM may have updates. Ensure you are using the latest available data files

2. **Station Coverage**: Remember that not all stations may be equally represented. Some stations may have better data quality than others

3. **Seasonal Patterns**: Results should account for the financial year structure (April to March) when exploring temporal trends

4. **Train Operating Companies (TOCs)**: The schedule data is based on published timetables. Actual operations may differ due to cancellations, rerouting, or service disruptions

5. **Incident Attribution**: Delays are attributed to specific incidents. Missing or misclassified incidents at unprocessed stations can affect delay propagation analysis

Reporting Issues and Getting Help
==================================

If you encounter issues not covered in this guide:

1. Check the project documentation in the ``docs/`` folder
2. Review error messages carefully for file path and naming issues
3. Verify that your data files match the format specified in the README
4. Contact the project maintainers at `ji-eun.byun@glasgow.ac.uk` with:
   - A description of the issue
   - The error message or unexpected behavior
   - Your preprocessing configuration (which stations/categories you processed)
   - The dates of data files you are using
