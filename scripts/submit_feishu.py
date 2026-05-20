"""Autofill the FlagOS Track 1 Feishu registration form.

Targets the Bitable shared form at:
  https://jwolpxeehx.feishu.cn/share/base/form/shrcnueKXpWaiDX4eZpVfJYtCjg

Field layout discovered by scripts/_feishu_inspect.py:
  idx 0  fldwjCjowp  text    Name
  idx 1  fldmGhJeWW  text    Country
  idx 2  fld7aQbc87  text    Location
  idx 3  fldrNh4jv2  text    Phone Number
  idx 4  fld2WBeIQD  text    Email Address
  idx 5  -           select  Track          (option 0 = Track 1)
  idx 6  fld9RJMEu5  text    Github ID
  idx 7  -           select  Platform       (option 1 = Kaggle)
  idx 8  -           select  Legal Policy   (option 0 = Agree)
  idx 9  -           select  Info Auth      (option 0 = Agree)
  idx 10 fldMcfmVk6  text    Referrer phone (optional, blank)

Run:
  python scripts/submit_feishu.py

After it fills, Chrome stays open. Visually verify, then click 提交 / Submit
in the browser. Return to the terminal and press Enter to close.

NOTE: personal info below is intentional and used only for this submission.
Do NOT commit changes to this file with the personal block populated.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

FORM_URL = "https://jwolpxeehx.feishu.cn/share/base/form/shrcnueKXpWaiDX4eZpVfJYtCjg"

YOUR_INFO: Dict[str, str] = {
    "name":     "YOUR NAME",
    "country":  "Country",
    "location": "City, Country",
    "phone":    "+000-000-0000",
    "email":    "you@example.com",
    "github":   "your-github-id",
    "referrer": "",
}

# (idx, field_id_or_None, type, value_or_option_index)
PLAN: List[tuple] = [
    (0,  "fldwjCjowp", "text",   YOUR_INFO["name"]),
    (1,  "fldmGhJeWW", "text",   YOUR_INFO["country"]),
    (2,  "fld7aQbc87", "text",   YOUR_INFO["location"]),
    (3,  "fldrNh4jv2", "text",   YOUR_INFO["phone"]),
    (4,  "fld2WBeIQD", "text",   YOUR_INFO["email"]),
    (5,  None,         "select", 0),   # Track 1
    (6,  "fld9RJMEu5", "text",   YOUR_INFO["github"]),
    (7,  None,         "select", 1),   # Kaggle
    (8,  None,         "select", 0),   # Agree
    (9,  None,         "select", 0),   # Agree
    (10, "fldMcfmVk6", "text",   YOUR_INFO["referrer"]),
]


def build_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--lang=en-US")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(options=opts)


def dismiss_overlays(driver: webdriver.Chrome) -> None:
    js = """
    document.querySelectorAll('button').forEach(b => {
      const t = (b.innerText || '').trim();
      if (['Got it', '知道了', '我知道了'].includes(t)) b.click();
    });
    """
    try:
        driver.execute_script(js)
    except Exception:
        pass


def scroll_form(driver: webdriver.Chrome) -> None:
    driver.execute_script("window.scrollTo(0, 0);")
    for _ in range(40):
        driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(0.2)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)


def fill_text(driver: webdriver.Chrome, item_idx: int, value: str) -> bool:
    if not value:
        return True
    js = """
    const items = document.querySelectorAll('.bitable-form-item');
    const n = items[arguments[0]];
    if (!n) return null;
    const editable = n.querySelector('[contenteditable="true"]');
    if (!editable) return null;
    editable.scrollIntoView({block: 'center'});
    return true;
    """
    ok = driver.execute_script(js, item_idx)
    if not ok:
        print(f"  [text idx={item_idx}] no contenteditable found")
        return False
    time.sleep(0.2)
    # Click to focus then send keys via ActionChains
    editable = driver.execute_script(
        "return document.querySelectorAll('.bitable-form-item')[arguments[0]]"
        ".querySelector('[contenteditable=\"true\"]');",
        item_idx,
    )
    try:
        editable.click()
        time.sleep(0.15)
        # Clear any zero-width content first
        editable.send_keys(Keys.CONTROL, "a")
        editable.send_keys(Keys.DELETE)
        time.sleep(0.05)
        editable.send_keys(value)
        time.sleep(0.15)
        print(f"  [text idx={item_idx}] filled: {value!r}")
        return True
    except Exception as exc:
        print(f"  [text idx={item_idx}] failed: {exc}")
        return False


def fill_select(driver: webdriver.Chrome, item_idx: int, option_idx: int) -> bool:
    js = """
    const items = document.querySelectorAll('.bitable-form-item');
    const n = items[arguments[0]];
    if (!n) return 'no-item';
    const rows = n.querySelectorAll('.base-component-select-list-editor-row');
    if (!rows.length) return 'no-rows';
    const r = rows[arguments[1]];
    if (!r) return 'no-row-idx';
    r.scrollIntoView({block: 'center'});
    const opt = r.querySelector('.base-component-select-list-editor-option') || r;
    opt.click();
    return 'clicked:' + rows.length;
    """
    res = driver.execute_script(js, item_idx, option_idx)
    print(f"  [select idx={item_idx} opt={option_idx}] {res}")
    return isinstance(res, str) and res.startswith("clicked")


def main() -> None:
    print(f"Opening {FORM_URL}")
    driver = build_driver()
    try:
        driver.get(FORM_URL)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".bitable-form-item"))
        )
        time.sleep(4)
        dismiss_overlays(driver)
        scroll_form(driver)
        time.sleep(1)

        # Confirm field count matches what we mapped
        count = driver.execute_script("return document.querySelectorAll('.bitable-form-item').length;")
        print(f"Detected {count} form items (plan has {len(PLAN)}).")
        if count != len(PLAN):
            print("WARNING: count mismatch. Continuing anyway.")

        print("\nFilling form ...")
        for item_idx, _fid, kind, value in PLAN:
            if kind == "text":
                fill_text(driver, item_idx, str(value))
            elif kind == "select":
                fill_select(driver, item_idx, int(value))
            time.sleep(0.25)

        scroll_form(driver)
        print(
            "\n"
            "============================================================\n"
            "  Auto-fill complete. Verify every field in Chrome,\n"
            "  then click the submit button (提交).\n"
            "  Press Enter here to close Chrome when done.\n"
            "============================================================\n"
        )
        input("Press Enter once you've clicked submit (or to abort): ")
    finally:
        driver.quit()
        print("Done.")


if __name__ == "__main__":
    main()
