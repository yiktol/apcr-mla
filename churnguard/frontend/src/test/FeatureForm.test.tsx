import { describe, expect, it } from "vitest";
import { validateFeatures } from "../components/FeatureForm";
import { DEFAULT_FEATURES } from "../features";

describe("validateFeatures", () => {
  it("accepts the real sample defaults", () => {
    expect(validateFeatures(DEFAULT_FEATURES)).toEqual({});
  });

  it("rejects an out-of-range tenure", () => {
    const errors = validateFeatures({ ...DEFAULT_FEATURES, tenure: 999 });
    expect(errors.tenure).toBeDefined();
  });

  it("rejects a non-integer tenure", () => {
    const errors = validateFeatures({ ...DEFAULT_FEATURES, tenure: 3.5 });
    expect(errors.tenure).toBeDefined();
  });

  it("rejects negative charges", () => {
    const errors = validateFeatures({ ...DEFAULT_FEATURES, MonthlyCharges: -1 });
    expect(errors.MonthlyCharges).toBeDefined();
  });
});
