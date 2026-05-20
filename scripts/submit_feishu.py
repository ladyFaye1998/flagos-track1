"""Selenium helper for the FlagOS Track 1 Feishu registration form.

The form lives at:
    https://jwolpxeehx.feishu.cn/share/base/form/shrcnueKXpWaiDX4eZpVfJYtCjg

Feishu is the Chinese product (Lark internationally). The form is heavily
JavaScript-rendered, so the script needs a real browser to interact with it.

Usage (Windows PowerShell):

    pip install selenium webdriver-manager
    # Fill in YOUR_INFO below with your real details, then:
    python scripts/submit_feishu.py

The script will:
  1. Open the form in Chrome.
  2. Wait for the SPA to finish loading.
  3. Enumerate every visible form field with its Chinese label
     and the inferred English meaning.
  4. Attempt to auto-fill based on the heuristic field-label match in
     ``FIELD_MAP`` below.
  5. STOP before clicking submit so you can verify in the browser window,
     then press Enter in the terminal to actually submit.

If the auto-fill mis-maps a field, you can edit it directly in the
browser window before pressing Enter.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:  # pragma: no cover
    print("Selenium is not installed. Run:  pip install selenium webdriver-manager")
    sys.exit(1)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAVE_WDM = True
except ImportError:
    HAVE_WDM = False


FORM_URL = "https://jwolpxeehx.feishu.cn/share/base/form/shrcnueKXpWaiDX4eZpVfJYtCjg"


# ---------------------------------------------------------------------------
# >>>>>>>>>>>>>>>>>>>>>>>  EDIT THIS BLOCK WITH YOUR INFO  <<<<<<<<<<<<<<<<<<<
# ---------------------------------------------------------------------------
YOUR_INFO: Dict[str, str] = {
    "name":           "Danielle Lesin",          # 姓名
    "phone":          "+972-XXX-XXX-XXXX",        # 手机号 (Phone)
    "email":          "ladyfaye1998@your.email",  # 邮箱 (Email)
    "team_name":      "FlagOS Track1 Solo",       # 团队名称 (Team name, OK if solo)
    "team_size":      "1",                        # 团队人数 (Team size, 1-3)
    "track":          "Track 1",                  # 赛道 / Track choice
    "github":         "https://github.com/ladyFaye1998/flagos-track1",
    "kaggle":         "https://www.kaggle.com/ladyfaye",
    "country":        "Israel",                   # 国家/地区
    "organization":   "Independent",              # 单位 / Organization
    "experience":     "Independent ML engineer; competed in legal-IR, March Madness, ArtSleuth and AIMO competitions on Kaggle.",
    "motivation":     "Open-source contribution to FlagGems via 20-operator Triton implementation under Apache-2.0.",
}
# ---------------------------------------------------------------------------


# Keyword → field key mapping. The script reads each field's visible Chinese
# label and tries to match a keyword here. Adjust if a field is mis-mapped.
FIELD_MAP: List[tuple] = [
    # (Chinese keyword(s) in the field label, key in YOUR_INFO above)
    (("姓名", "Name", "name"),                "name"),
    (("电话", "手机", "Phone", "Mobile"),     "phone"),
    (("邮箱", "Email", "邮件"),               "email"),
    (("团队名", "Team name", "队伍"),         "team_name"),
    (("人数", "size", "成员"),                "team_size"),
    (("赛道", "Track", "track"),              "track"),
    (("GitHub", "github", "代码仓"),          "github"),
    (("Kaggle", "kaggle"),                    "kaggle"),
    (("国家", "Country", "地区"),             "country"),
    (("单位", "组织", "Organization", "公司"),"organization"),
    (("经验", "Experience", "背景"),          "experience"),
    (("动机", "Motivation", "意愿"),          "motivation"),
]


def build_driver() -> webdriver.Chrome:
    """Boot a Chrome WebDriver instance with sane defaults."""
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--lang=en-US")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if HAVE_WDM:
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)
    return webdriver.Chrome(options=opts)  # relies on system chromedriver


def match_field_key(label_text: str) -> str | None:
    label_lower = label_text.lower()
    for keywords, key in FIELD_MAP:
        for kw in keywords:
            if kw.lower() in label_lower:
                return key
    return None


def slow_type(el, text: str, delay: float = 0.04) -> None:
    el.clear()
    for ch in text:
        el.send_keys(ch)
        time.sleep(delay)


def enumerate_and_fill(driver: webdriver.Chrome) -> None:
    """Find all visible form-row containers, print them, and try to fill."""
    # Feishu form items typically live in elements with role="textbox" or
    # input/textarea inside a labelled container.
    inputs = driver.find_elements(By.CSS_SELECTOR, "input, textarea")
    visible = [el for el in inputs if el.is_displayed() and el.is_enabled()]
    print(f"\nFound {len(visible)} interactive form fields.\n")

    for idx, el in enumerate(visible, 1):
        # Walk up to find the nearest visible label.
        label = ""
        try:
            label = el.find_element(
                By.XPATH,
                "ancestor::*[contains(@class,'form-item')][1]"
                "//*[contains(@class,'label') or contains(@class,'title')][1]",
            ).text.strip()
        except Exception:
            try:
                label = el.find_element(By.XPATH, "preceding::label[1]").text.strip()
            except Exception:
                pass

        key = match_field_key(label) if label else None
        action = "(no auto-fill match; please fill manually)"
        if key and key in YOUR_INFO:
            try:
                slow_type(el, YOUR_INFO[key])
                action = f"-> filled with YOUR_INFO['{key}']"
            except Exception as exc:
                action = f"-> error filling: {exc}"

        snippet = label[:40].replace("\n", " ") if label else "(no label found)"
        print(f"  [{idx:02d}]  {snippet:<42}  {action}")


def main() -> None:
    if "XXX-XXX" in YOUR_INFO["phone"] or "your.email" in YOUR_INFO["email"]:
        print(
            "ERROR: edit scripts/submit_feishu.py and fill in YOUR_INFO with your "
            "real details before running."
        )
        sys.exit(2)

    print(f"Opening Feishu form: {FORM_URL}")
    driver = build_driver()
    try:
        driver.get(FORM_URL)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input, textarea"))
        )
        print("Form loaded. Waiting 5s for late-rendered fields...")
        time.sleep(5)

        enumerate_and_fill(driver)

        print(
            "\n"
            "============================================================\n"
            "  AUTO-FILL DONE. The browser window is still open.\n"
            "  1) Visually check every field in the form.\n"
            "  2) Fix anything that looks wrong directly in the browser.\n"
            "  3) When ready, click the SUBMIT button in the browser\n"
            "     (Chinese: 提交 / 'Tijiao').\n"
            "  4) Then press Enter here to close the browser.\n"
            "============================================================\n"
        )
        input("Press Enter once you have clicked submit in the browser... ")
    finally:
        driver.quit()
        print("Done.")


if __name__ == "__main__":
    main()
