# Data Labeling Guide: Production Drawing Comparison

## 1. Objective

Your task is to find and document differences between two sets of production drawings:
- **`toProduction`**: The drawings our team sent to the production facility.
- **`fromProduction`**: The drawings the production facility created based on our plans.

You will identify pairs of matching panels (one from each set) and label them as either a **match (1)** or a **mismatch (0)** based on whether their dimensions are identical. The final goal is to create a balanced dataset: for every mismatch you find, you must also find a corresponding pair that is correct.

---

## 2. Step-by-Step Workflow

### Step 1: Convert PDFs to High-Quality Images
Before you begin, you must convert all relevant pages from both `toProduction` and `fromProduction` PDF files into images.

- **Tool**: Use any standard PDF-to-image converter (e.g., Adobe Acrobat, online tools).
- **Settings**:
    - **Resolution**: **300 DPI** is critical for clarity.
    - **Paper Size**: Export the image using the PDF's original paper size (e.g., A1, 24x36 inches). Do not "fit to page" on a different paper size.
- **Format**: Save images as `.jpg` or `.png`.

### Step 2: Organize Your Files
Create the following folder structure to store your work:

```
Labeling_Project/
├── toProduction_images/
│   └── (all image files from toProduction PDFs go here)
│
├── fromProduction_images/
│   └── (all image files from fromProduction PDFs go here)
│
└── labels.xlsx
```

- **Naming Convention**: Name your image files clearly so you can identify them later. A good format is `{Original_PDF_Name}_page_{Page_Number}.jpg`.
  - *Example*: `ProjectA_Sheet1_toProduction_page_01.jpg`

### Step 3: Find and Label Pairs
Your main task is to compare the images.

1.  **Identify Matching Panels**: Look for panels that have the **same panel tag** (e.g., "P-105") in both a `toProduction` drawing and a `fromProduction` drawing. This pair is your subject.

2.  **Look for Mismatches (Label 0)**:
    - Carefully compare the dimensions listed on the panel in the `toProduction` image against the `fromProduction` image.
    - If you find **any difference** in the dimensions (a number is different, a dimension is missing, etc.), this is a **mismatch**.
    - In your Excel file (`labels.xlsx`), create a new row.
    - In the `toProduction` column, enter the filename of the `toProduction` image.
    - In the `fromProduction` column, enter the filename of the `fromProduction` image.
    - In the `label` column, enter **0**.

3.  **Find a Corresponding Match (Label 1)**:
    - For **every mismatch (0) you find**, you must then find another pair of panels (with a different panel tag) that are a **perfect match**.
    - A perfect match means all dimensions listed on the `toProduction` panel are identical to the `fromProduction` panel.
    - Add this matching pair as a new row in your Excel file and set the `label` to **1**. This ensures our final dataset is balanced.

---

## 3. The `labels.xlsx` File

Your final output will be a single Excel file with three columns. It should look like this:

| toProduction                               | fromProduction                             | label |
| ------------------------------------------ | ------------------------------------------ | ----- |
| `ProjectA_toProd_p01.jpg`                  | `ProjectA_fromProd_p05.jpg`                | 0     |
| `ProjectA_toProd_p02.jpg`                  | `ProjectA_fromProd_p08.jpg`                | 1     |
| `ProjectB_toProd_p04.jpg`                  | `ProjectB_fromProd_p02.jpg`                | 0     |
| `ProjectB_toProd_p04.jpg`                  | `ProjectB_fromProd_p03.jpg`                | 1     |
| ...                                        | ...                                        | ...   |

- **`toProduction` column**: The exact filename of the image from the `toProduction_images` folder.
- **`fromProduction` column**: The exact filename of the image from the `fromProduction_images` folder.
- **`label` column**: **0** for a mismatch in dimensions, **1** for a perfect match.

---

## 4. Final Deliverables

When you are finished, please provide a zip file containing:
1.  The `toProduction_images` folder with all the image files you used.
2.  The `fromProduction_images` folder with all the image files you used.
3.  The final `labels.xlsx` file.

Thank you!
