
# Types of Tests on SAM (Segment Anything Model)

The evaluation of SAM has been organized into various approaches, grouped by category. Below is a detailed explanation of the main strategies employed:

---

## 1. **Unsegmented Fill Strategy**

When SAM segments an image, it may leave certain areas between object instances unsegmented. Several strategies were explored to fill these gaps:

- **ERD (Hereditary):**
  - Unsegmented pixels are filled using values from the previous pseudo-labels.
  - This is useful for preserving label consistency in unchanged areas.

- **MAX:**
  - Treats black (unsegmented) zones as connected components and assigns them the majority label.
  - Downside: SAM often leaves borders unsegmented, leading to more harm than good when this strategy is applied.

---

## 2. **Centroid Computation Strategy**

- **Normal:**
  - The centroid of the component is computed and shifted to the nearest useful region based on the pseudo-label.

- **IB (Inner Box):**
  - Computes the largest rectangle that contains only pixels of the intended label.
  - While this method is computationally expensive, it is robust and accurate—though it does not offer major performance improvements.

---

## 3. **Region Selection for Segmentation**

- **Normal:**
  - Components of a label are extracted, and noise is removed using a dilated mask.
  - Effective, but struggles with "Picasso-style" images (many label fragments), which can lead to excessive computation and mask degradation.

- **NS (No Small):**
  - Only labels with a pixel count above a set percentage of the image size are considered.
  - This offers a crude solution to the noise problem in pseudo-labels.

- **NM (No Multiple):**
  - Excludes components that overlap with others during segmentation.
  - Downside: if a wrongly segmented component covers the entire image, it invalidates the result.

---

## 4. **Label Exclusion Strategy**

- **No exclusion:**
  - All labels, including critical ones (e.g., walls, floors), are considered.
  - SAM refines boundaries with support from adjacent labels.
  - Often effective because critical labels are processed early and overwritten if errors occur.

- **nW (No wall) / nWF (No wall-floor):**
  - Excludes risky labels such as walls and floors.
  - Reduces risk of critical errors but can lead to less accurate boundary delineation.
  - Tests suggest this is not generally beneficial.

---

## 5. **SAM Prompt Strategy (Captioning)**

- **Normal:**
  - Both bounding box and centroid are used as prompts for SAM.

- **OBB (Only Bounding Box):**
  - SAM segments based on the object’s overall bounding box.

- **OP (Only Points):**
  - Not tested, known to often misinterpret the target object.

---

## 6. **Fill Strategy (Majority Label Assignment)**

The fill strategy determines how each segmented mask is assigned a semantic label.

- **Normal (Majority):**
  - Each SAM-generated mask is labeled using the most frequent (majority) class present in the ground truth overlap.
  - Simple but can propagate errors if masks include multiple objects.

- **nWm (No Wall Max) / nWFm (No Wall-Floor Max):**
  - For selected critical classes, majority voting is avoided.
  - Instead, a predefined label is enforced to mitigate damage caused by segmentation errors.
  - Helps preserve important classes that may be small or error-prone.
