#!/usr/bin/env python3
"""Generate Auditor.xcodeproj (Xcode 16 format) for the 7oda Auditor app.

The project uses file-system synchronized groups (objectVersion 77): every file
under ``Auditor/`` belongs to the app target and every file under
``AuditorTests/`` to the hosted unit-test target, so adding a Swift file needs
no project change. The core logic comes from the local Swift package
``AuditorCore/``.

Object identifiers are derived from fixed labels, so running the script again
produces a byte-identical project. Run from anywhere:

    python3 7oda-Auditor/tools/gen_xcodeproj.py

Requires Xcode 16 or newer to open.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "Auditor.xcodeproj"

APP = "Auditor"
TESTS = "AuditorTests"
PACKAGE_DIR = "AuditorCore"
PACKAGE_PRODUCT = "AuditorCore"
BUNDLE_ID = "com.mahmoudalyosify.auditor"
DISPLAY_NAME = "7oda Auditor"
MARKETING_VERSION = "0.1.0"
BUILD_NUMBER = "1"
IOS_TARGET = "17.0"
MOTION_TEXT = ("7oda Auditor reads the motion sensors while the phone vibrates, to recognise "
               "the hand that holds it. Nothing leaves the phone.")


def oid(label: str) -> str:
    return hashlib.md5(f"7oda-auditor:{label}".encode()).hexdigest()[:24].upper()


ID = {name: oid(name) for name in [
    "project", "mainGroup", "productsGroup", "appGroup", "testsGroup",
    "appProduct", "testsProduct",
    "appTarget", "testsTarget",
    "appSources", "appFrameworks", "appResources",
    "testsSources", "testsFrameworks", "testsResources",
    "pkgRef", "pkgProduct", "pkgBuildFile",
    "proxy", "targetDependency",
    "projectConfigList", "projectDebug", "projectRelease",
    "appConfigList", "appDebug", "appRelease",
    "testsConfigList", "testsDebug", "testsRelease",
]}


def q(value: str) -> str:
    """Quote a value the way Xcode does when it contains special characters."""
    if value and all(c.isalnum() or c in "._/" for c in value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def settings_block(settings: dict, indent: str) -> str:
    lines = []
    for key in sorted(settings):
        value = settings[key]
        if isinstance(value, list):
            inner = "".join(f"{indent}\t{q(v)},\n" for v in value)
            lines.append(f"{indent}{key} = (\n{inner}{indent});\n")
        else:
            lines.append(f"{indent}{key} = {q(str(value))};\n")
    return "".join(lines)


PROJECT_COMMON = {
    "ALWAYS_SEARCH_USER_PATHS": "NO",
    "ASSETCATALOG_COMPILER_GENERATE_SWIFT_ASSET_SYMBOL_EXTENSIONS": "YES",
    "CLANG_ANALYZER_NONNULL": "YES",
    "CLANG_ANALYZER_NUMBER_OBJECT_CONVERSION": "YES_AGGRESSIVE",
    "CLANG_CXX_LANGUAGE_STANDARD": "gnu++20",
    "CLANG_ENABLE_MODULES": "YES",
    "CLANG_ENABLE_OBJC_ARC": "YES",
    "CLANG_ENABLE_OBJC_WEAK": "YES",
    "CLANG_WARN_BLOCK_CAPTURE_AUTORELEASING": "YES",
    "CLANG_WARN_BOOL_CONVERSION": "YES",
    "CLANG_WARN_COMMA": "YES",
    "CLANG_WARN_CONSTANT_CONVERSION": "YES",
    "CLANG_WARN_DEPRECATED_OBJC_IMPLEMENTATIONS": "YES",
    "CLANG_WARN_DIRECT_OBJC_ISA_USAGE": "YES_ERROR",
    "CLANG_WARN_DOCUMENTATION_COMMENTS": "YES",
    "CLANG_WARN_EMPTY_BODY": "YES",
    "CLANG_WARN_ENUM_CONVERSION": "YES",
    "CLANG_WARN_INFINITE_RECURSION": "YES",
    "CLANG_WARN_INT_CONVERSION": "YES",
    "CLANG_WARN_NON_LITERAL_NULL_CONVERSION": "YES",
    "CLANG_WARN_OBJC_IMPLICIT_RETAIN_SELF": "YES",
    "CLANG_WARN_OBJC_LITERAL_CONVERSION": "YES",
    "CLANG_WARN_OBJC_ROOT_CLASS": "YES_ERROR",
    "CLANG_WARN_QUOTED_INCLUDE_IN_FRAMEWORK_HEADER": "YES",
    "CLANG_WARN_RANGE_LOOP_ANALYSIS": "YES",
    "CLANG_WARN_STRICT_PROTOTYPES": "YES",
    "CLANG_WARN_SUSPICIOUS_MOVE": "YES",
    "CLANG_WARN_UNGUARDED_AVAILABILITY": "YES_AGGRESSIVE",
    "CLANG_WARN_UNREACHABLE_CODE": "YES",
    "CLANG_WARN__DUPLICATE_METHOD_MATCH": "YES",
    "COPY_PHASE_STRIP": "NO",
    "ENABLE_STRICT_OBJC_MSGSEND": "YES",
    "ENABLE_USER_SCRIPT_SANDBOXING": "YES",
    "GCC_C_LANGUAGE_STANDARD": "gnu17",
    "GCC_NO_COMMON_BLOCKS": "YES",
    "GCC_WARN_64_TO_32_BIT_CONVERSION": "YES",
    "GCC_WARN_ABOUT_RETURN_TYPE": "YES_ERROR",
    "GCC_WARN_UNDECLARED_SELECTOR": "YES",
    "GCC_WARN_UNINITIALIZED_AUTOS": "YES_AGGRESSIVE",
    "GCC_WARN_UNUSED_FUNCTION": "YES",
    "GCC_WARN_UNUSED_VARIABLE": "YES",
    "IPHONEOS_DEPLOYMENT_TARGET": IOS_TARGET,
    "LOCALIZATION_PREFERS_STRING_CATALOGS": "YES",
    "MTL_FAST_MATH": "YES",
    "SDKROOT": "iphoneos",
    "SWIFT_VERSION": "5.0",
}
PROJECT_DEBUG = {
    **PROJECT_COMMON,
    "DEBUG_INFORMATION_FORMAT": "dwarf",
    "ENABLE_TESTABILITY": "YES",
    "GCC_DYNAMIC_NO_PIC": "NO",
    "GCC_OPTIMIZATION_LEVEL": "0",
    "GCC_PREPROCESSOR_DEFINITIONS": ["DEBUG=1", "$(inherited)"],
    "MTL_ENABLE_DEBUG_INFO": "INCLUDE_SOURCE",
    "ONLY_ACTIVE_ARCH": "YES",
    "SWIFT_ACTIVE_COMPILATION_CONDITIONS": "DEBUG $(inherited)",
    "SWIFT_OPTIMIZATION_LEVEL": "-Onone",
}
PROJECT_RELEASE = {
    **PROJECT_COMMON,
    "DEBUG_INFORMATION_FORMAT": "dwarf-with-dsym",
    "ENABLE_NS_ASSERTIONS": "NO",
    "MTL_ENABLE_DEBUG_INFO": "NO",
    "SWIFT_COMPILATION_MODE": "wholemodule",
    "VALIDATE_PRODUCT": "YES",
}
APP_SETTINGS = {
    "ASSETCATALOG_COMPILER_APPICON_NAME": "AppIcon",
    "ASSETCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME": "AccentColor",
    "CODE_SIGN_STYLE": "Automatic",
    "CURRENT_PROJECT_VERSION": BUILD_NUMBER,
    "DEVELOPMENT_TEAM": "",
    "ENABLE_PREVIEWS": "YES",
    "GENERATE_INFOPLIST_FILE": "YES",
    "INFOPLIST_KEY_CFBundleDisplayName": DISPLAY_NAME,
    "INFOPLIST_KEY_ITSAppUsesNonExemptEncryption": "NO",
    "INFOPLIST_KEY_LSApplicationCategoryType": "public.app-category.utilities",
    "INFOPLIST_KEY_NSMotionUsageDescription": MOTION_TEXT,
    "INFOPLIST_KEY_UIApplicationSceneManifest_Generation": "YES",
    "INFOPLIST_KEY_UIApplicationSupportsIndirectInputEvents": "YES",
    "INFOPLIST_KEY_UILaunchScreen_Generation": "YES",
    "INFOPLIST_KEY_UIRequiresFullScreen": "YES",
    "INFOPLIST_KEY_UISupportedInterfaceOrientations": "UIInterfaceOrientationPortrait",
    "INFOPLIST_KEY_UISupportedInterfaceOrientations_iPhone": "UIInterfaceOrientationPortrait",
    "IPHONEOS_DEPLOYMENT_TARGET": IOS_TARGET,
    "LD_RUNPATH_SEARCH_PATHS": ["$(inherited)", "@executable_path/Frameworks"],
    "MARKETING_VERSION": MARKETING_VERSION,
    "PRODUCT_BUNDLE_IDENTIFIER": BUNDLE_ID,
    "PRODUCT_NAME": "$(TARGET_NAME)",
    "SUPPORTED_PLATFORMS": "iphoneos iphonesimulator",
    "SUPPORTS_MACCATALYST": "NO",
    "SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD": "NO",
    "SUPPORTS_XR_DESIGNED_FOR_IPHONE_IPAD": "NO",
    "SWIFT_EMIT_LOC_STRINGS": "YES",
    "SWIFT_VERSION": "5.0",
    "TARGETED_DEVICE_FAMILY": "1",
}
TEST_SETTINGS = {
    "BUNDLE_LOADER": "$(TEST_HOST)",
    "CODE_SIGN_STYLE": "Automatic",
    "CURRENT_PROJECT_VERSION": BUILD_NUMBER,
    "DEVELOPMENT_TEAM": "",
    "GENERATE_INFOPLIST_FILE": "YES",
    "IPHONEOS_DEPLOYMENT_TARGET": IOS_TARGET,
    "MARKETING_VERSION": MARKETING_VERSION,
    "PRODUCT_BUNDLE_IDENTIFIER": f"{BUNDLE_ID}.tests",
    "PRODUCT_NAME": "$(TARGET_NAME)",
    "SUPPORTED_PLATFORMS": "iphoneos iphonesimulator",
    "SUPPORTS_MACCATALYST": "NO",
    "SWIFT_EMIT_LOC_STRINGS": "NO",
    "SWIFT_VERSION": "5.0",
    "TARGETED_DEVICE_FAMILY": "1",
    "TEST_HOST": f"$(BUILT_PRODUCTS_DIR)/{APP}.app/$(BUNDLE_EXECUTABLE_FOLDER_PATH)/{APP}",
}


def build_configuration(key: str, name: str, settings: dict) -> str:
    return (f"\t\t{ID[key]} /* {name} */ = {{\n"
            f"\t\t\tisa = XCBuildConfiguration;\n"
            f"\t\t\tbuildSettings = {{\n"
            f"{settings_block(settings, chr(9) * 4)}"
            f"\t\t\t}};\n"
            f"\t\t\tname = {name};\n"
            f"\t\t}};\n")


def pbxproj() -> str:
    i = ID
    out: list[str] = []
    w = out.append
    w("// !$*UTF8*$!\n{\n\tarchiveVersion = 1;\n\tclasses = {\n\t};\n\tobjectVersion = 77;\n\tobjects = {\n\n")

    w("/* Begin PBXBuildFile section */\n")
    w(f"\t\t{i['pkgBuildFile']} /* {PACKAGE_PRODUCT} in Frameworks */ = {{isa = PBXBuildFile; productRef = {i['pkgProduct']} /* {PACKAGE_PRODUCT} */; }};\n")
    w("/* End PBXBuildFile section */\n\n")

    w("/* Begin PBXContainerItemProxy section */\n")
    w(f"\t\t{i['proxy']} /* PBXContainerItemProxy */ = {{\n"
      f"\t\t\tisa = PBXContainerItemProxy;\n"
      f"\t\t\tcontainerPortal = {i['project']} /* Project object */;\n"
      f"\t\t\tproxyType = 1;\n"
      f"\t\t\tremoteGlobalIDString = {i['appTarget']};\n"
      f"\t\t\tremoteInfo = {APP};\n"
      f"\t\t}};\n")
    w("/* End PBXContainerItemProxy section */\n\n")

    w("/* Begin PBXFileReference section */\n")
    w(f"\t\t{i['appProduct']} /* {APP}.app */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = {APP}.app; sourceTree = BUILT_PRODUCTS_DIR; }};\n")
    w(f"\t\t{i['testsProduct']} /* {TESTS}.xctest */ = {{isa = PBXFileReference; explicitFileType = wrapper.cfbundle; includeInIndex = 0; path = {TESTS}.xctest; sourceTree = BUILT_PRODUCTS_DIR; }};\n")
    w("/* End PBXFileReference section */\n\n")

    w("/* Begin PBXFileSystemSynchronizedRootGroup section */\n")
    for key, path in (("appGroup", APP), ("testsGroup", TESTS)):
        w(f"\t\t{i[key]} /* {path} */ = {{\n"
          f"\t\t\tisa = PBXFileSystemSynchronizedRootGroup;\n"
          f"\t\t\tpath = {path};\n"
          f"\t\t\tsourceTree = \"<group>\";\n"
          f"\t\t}};\n")
    w("/* End PBXFileSystemSynchronizedRootGroup section */\n\n")

    w("/* Begin PBXFrameworksBuildPhase section */\n")
    w(f"\t\t{i['appFrameworks']} /* Frameworks */ = {{\n"
      f"\t\t\tisa = PBXFrameworksBuildPhase;\n"
      f"\t\t\tbuildActionMask = 2147483647;\n"
      f"\t\t\tfiles = (\n"
      f"\t\t\t\t{i['pkgBuildFile']} /* {PACKAGE_PRODUCT} in Frameworks */,\n"
      f"\t\t\t);\n"
      f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
      f"\t\t}};\n")
    w(f"\t\t{i['testsFrameworks']} /* Frameworks */ = {{\n"
      f"\t\t\tisa = PBXFrameworksBuildPhase;\n"
      f"\t\t\tbuildActionMask = 2147483647;\n"
      f"\t\t\tfiles = (\n"
      f"\t\t\t);\n"
      f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
      f"\t\t}};\n")
    w("/* End PBXFrameworksBuildPhase section */\n\n")

    w("/* Begin PBXGroup section */\n")
    w(f"\t\t{i['mainGroup']} = {{\n"
      f"\t\t\tisa = PBXGroup;\n"
      f"\t\t\tchildren = (\n"
      f"\t\t\t\t{i['appGroup']} /* {APP} */,\n"
      f"\t\t\t\t{i['testsGroup']} /* {TESTS} */,\n"
      f"\t\t\t\t{i['productsGroup']} /* Products */,\n"
      f"\t\t\t);\n"
      f"\t\t\tsourceTree = \"<group>\";\n"
      f"\t\t}};\n")
    w(f"\t\t{i['productsGroup']} /* Products */ = {{\n"
      f"\t\t\tisa = PBXGroup;\n"
      f"\t\t\tchildren = (\n"
      f"\t\t\t\t{i['appProduct']} /* {APP}.app */,\n"
      f"\t\t\t\t{i['testsProduct']} /* {TESTS}.xctest */,\n"
      f"\t\t\t);\n"
      f"\t\t\tname = Products;\n"
      f"\t\t\tsourceTree = \"<group>\";\n"
      f"\t\t}};\n")
    w("/* End PBXGroup section */\n\n")

    w("/* Begin PBXNativeTarget section */\n")
    w(f"\t\t{i['appTarget']} /* {APP} */ = {{\n"
      f"\t\t\tisa = PBXNativeTarget;\n"
      f"\t\t\tbuildConfigurationList = {i['appConfigList']} /* Build configuration list for PBXNativeTarget \"{APP}\" */;\n"
      f"\t\t\tbuildPhases = (\n"
      f"\t\t\t\t{i['appSources']} /* Sources */,\n"
      f"\t\t\t\t{i['appFrameworks']} /* Frameworks */,\n"
      f"\t\t\t\t{i['appResources']} /* Resources */,\n"
      f"\t\t\t);\n"
      f"\t\t\tbuildRules = (\n"
      f"\t\t\t);\n"
      f"\t\t\tdependencies = (\n"
      f"\t\t\t);\n"
      f"\t\t\tfileSystemSynchronizedGroups = (\n"
      f"\t\t\t\t{i['appGroup']} /* {APP} */,\n"
      f"\t\t\t);\n"
      f"\t\t\tname = {APP};\n"
      f"\t\t\tpackageProductDependencies = (\n"
      f"\t\t\t\t{i['pkgProduct']} /* {PACKAGE_PRODUCT} */,\n"
      f"\t\t\t);\n"
      f"\t\t\tproductName = {APP};\n"
      f"\t\t\tproductReference = {i['appProduct']} /* {APP}.app */;\n"
      f"\t\t\tproductType = \"com.apple.product-type.application\";\n"
      f"\t\t}};\n")
    w(f"\t\t{i['testsTarget']} /* {TESTS} */ = {{\n"
      f"\t\t\tisa = PBXNativeTarget;\n"
      f"\t\t\tbuildConfigurationList = {i['testsConfigList']} /* Build configuration list for PBXNativeTarget \"{TESTS}\" */;\n"
      f"\t\t\tbuildPhases = (\n"
      f"\t\t\t\t{i['testsSources']} /* Sources */,\n"
      f"\t\t\t\t{i['testsFrameworks']} /* Frameworks */,\n"
      f"\t\t\t\t{i['testsResources']} /* Resources */,\n"
      f"\t\t\t);\n"
      f"\t\t\tbuildRules = (\n"
      f"\t\t\t);\n"
      f"\t\t\tdependencies = (\n"
      f"\t\t\t\t{i['targetDependency']} /* PBXTargetDependency */,\n"
      f"\t\t\t);\n"
      f"\t\t\tfileSystemSynchronizedGroups = (\n"
      f"\t\t\t\t{i['testsGroup']} /* {TESTS} */,\n"
      f"\t\t\t);\n"
      f"\t\t\tname = {TESTS};\n"
      f"\t\t\tpackageProductDependencies = (\n"
      f"\t\t\t);\n"
      f"\t\t\tproductName = {TESTS};\n"
      f"\t\t\tproductReference = {i['testsProduct']} /* {TESTS}.xctest */;\n"
      f"\t\t\tproductType = \"com.apple.product-type.bundle.unit-test\";\n"
      f"\t\t}};\n")
    w("/* End PBXNativeTarget section */\n\n")

    w("/* Begin PBXProject section */\n")
    w(f"\t\t{i['project']} /* Project object */ = {{\n"
      f"\t\t\tisa = PBXProject;\n"
      f"\t\t\tattributes = {{\n"
      f"\t\t\t\tBuildIndependentTargetsInParallel = 1;\n"
      f"\t\t\t\tLastSwiftUpdateCheck = 1600;\n"
      f"\t\t\t\tLastUpgradeCheck = 1600;\n"
      f"\t\t\t\tTargetAttributes = {{\n"
      f"\t\t\t\t\t{i['appTarget']} = {{\n"
      f"\t\t\t\t\t\tCreatedOnToolsVersion = 16.0;\n"
      f"\t\t\t\t\t}};\n"
      f"\t\t\t\t\t{i['testsTarget']} = {{\n"
      f"\t\t\t\t\t\tCreatedOnToolsVersion = 16.0;\n"
      f"\t\t\t\t\t\tTestTargetID = {i['appTarget']};\n"
      f"\t\t\t\t\t}};\n"
      f"\t\t\t\t}};\n"
      f"\t\t\t}};\n"
      f"\t\t\tbuildConfigurationList = {i['projectConfigList']} /* Build configuration list for PBXProject \"{APP}\" */;\n"
      f"\t\t\tdevelopmentRegion = en;\n"
      f"\t\t\thasScannedForEncodings = 0;\n"
      f"\t\t\tknownRegions = (\n"
      f"\t\t\t\ten,\n"
      f"\t\t\t\tBase,\n"
      f"\t\t\t);\n"
      f"\t\t\tmainGroup = {i['mainGroup']};\n"
      f"\t\t\tminimizedProjectReferenceProxies = 1;\n"
      f"\t\t\tpackageReferences = (\n"
      f"\t\t\t\t{i['pkgRef']} /* XCLocalSwiftPackageReference \"{PACKAGE_DIR}\" */,\n"
      f"\t\t\t);\n"
      f"\t\t\tpreferredProjectObjectVersion = 77;\n"
      f"\t\t\tproductRefGroup = {i['productsGroup']} /* Products */;\n"
      f"\t\t\tprojectDirPath = \"\";\n"
      f"\t\t\tprojectRoot = \"\";\n"
      f"\t\t\ttargets = (\n"
      f"\t\t\t\t{i['appTarget']} /* {APP} */,\n"
      f"\t\t\t\t{i['testsTarget']} /* {TESTS} */,\n"
      f"\t\t\t);\n"
      f"\t\t}};\n")
    w("/* End PBXProject section */\n\n")

    w("/* Begin PBXResourcesBuildPhase section */\n")
    for key in ("appResources", "testsResources"):
        w(f"\t\t{i[key]} /* Resources */ = {{\n"
          f"\t\t\tisa = PBXResourcesBuildPhase;\n"
          f"\t\t\tbuildActionMask = 2147483647;\n"
          f"\t\t\tfiles = (\n"
          f"\t\t\t);\n"
          f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
          f"\t\t}};\n")
    w("/* End PBXResourcesBuildPhase section */\n\n")

    w("/* Begin PBXSourcesBuildPhase section */\n")
    for key in ("appSources", "testsSources"):
        w(f"\t\t{i[key]} /* Sources */ = {{\n"
          f"\t\t\tisa = PBXSourcesBuildPhase;\n"
          f"\t\t\tbuildActionMask = 2147483647;\n"
          f"\t\t\tfiles = (\n"
          f"\t\t\t);\n"
          f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
          f"\t\t}};\n")
    w("/* End PBXSourcesBuildPhase section */\n\n")

    w("/* Begin PBXTargetDependency section */\n")
    w(f"\t\t{i['targetDependency']} /* PBXTargetDependency */ = {{\n"
      f"\t\t\tisa = PBXTargetDependency;\n"
      f"\t\t\ttarget = {i['appTarget']} /* {APP} */;\n"
      f"\t\t\ttargetProxy = {i['proxy']} /* PBXContainerItemProxy */;\n"
      f"\t\t}};\n")
    w("/* End PBXTargetDependency section */\n\n")

    w("/* Begin XCBuildConfiguration section */\n")
    w(build_configuration("projectDebug", "Debug", PROJECT_DEBUG))
    w(build_configuration("projectRelease", "Release", PROJECT_RELEASE))
    w(build_configuration("appDebug", "Debug", APP_SETTINGS))
    w(build_configuration("appRelease", "Release", APP_SETTINGS))
    w(build_configuration("testsDebug", "Debug", TEST_SETTINGS))
    w(build_configuration("testsRelease", "Release", TEST_SETTINGS))
    w("/* End XCBuildConfiguration section */\n\n")

    w("/* Begin XCConfigurationList section */\n")
    for key, kind, name, dbg, rel in (
        ("projectConfigList", "PBXProject", APP, "projectDebug", "projectRelease"),
        ("appConfigList", "PBXNativeTarget", APP, "appDebug", "appRelease"),
        ("testsConfigList", "PBXNativeTarget", TESTS, "testsDebug", "testsRelease"),
    ):
        w(f"\t\t{i[key]} /* Build configuration list for {kind} \"{name}\" */ = {{\n"
          f"\t\t\tisa = XCConfigurationList;\n"
          f"\t\t\tbuildConfigurations = (\n"
          f"\t\t\t\t{i[dbg]} /* Debug */,\n"
          f"\t\t\t\t{i[rel]} /* Release */,\n"
          f"\t\t\t);\n"
          f"\t\t\tdefaultConfigurationIsVisible = 0;\n"
          f"\t\t\tdefaultConfigurationName = Release;\n"
          f"\t\t}};\n")
    w("/* End XCConfigurationList section */\n\n")

    w("/* Begin XCLocalSwiftPackageReference section */\n")
    w(f"\t\t{i['pkgRef']} /* XCLocalSwiftPackageReference \"{PACKAGE_DIR}\" */ = {{\n"
      f"\t\t\tisa = XCLocalSwiftPackageReference;\n"
      f"\t\t\trelativePath = {PACKAGE_DIR};\n"
      f"\t\t}};\n")
    w("/* End XCLocalSwiftPackageReference section */\n\n")

    w("/* Begin XCSwiftPackageProductDependency section */\n")
    w(f"\t\t{i['pkgProduct']} /* {PACKAGE_PRODUCT} */ = {{\n"
      f"\t\t\tisa = XCSwiftPackageProductDependency;\n"
      f"\t\t\tproductName = {PACKAGE_PRODUCT};\n"
      f"\t\t}};\n")
    w("/* End XCSwiftPackageProductDependency section */\n")

    w(f"\t}};\n\trootObject = {i['project']} /* Project object */;\n}}\n")
    return "".join(out)


def buildable(target_key: str, product: str, name: str) -> str:
    return (f'            <BuildableReference\n'
            f'               BuildableIdentifier = "primary"\n'
            f'               BlueprintIdentifier = "{ID[target_key]}"\n'
            f'               BuildableName = "{product}"\n'
            f'               BlueprintName = "{name}"\n'
            f'               ReferencedContainer = "container:{APP}.xcodeproj">\n'
            f'            </BuildableReference>\n')


def scheme() -> str:
    app_ref = buildable("appTarget", f"{APP}.app", APP)
    test_ref = buildable("testsTarget", f"{TESTS}.xctest", TESTS)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme
   LastUpgradeVersion = "1600"
   version = "1.7">
   <BuildAction
      parallelizeBuildables = "YES"
      buildImplicitDependencies = "YES"
      buildArchitectures = "Automatic">
      <BuildActionEntries>
         <BuildActionEntry
            buildForTesting = "YES"
            buildForRunning = "YES"
            buildForProfiling = "YES"
            buildForArchiving = "YES"
            buildForAnalyzing = "YES">
{app_ref}         </BuildActionEntry>
      </BuildActionEntries>
   </BuildAction>
   <TestAction
      buildConfiguration = "Debug"
      selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB"
      selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB"
      shouldUseLaunchSchemeArgsEnv = "YES"
      shouldAutocreateTestPlan = "YES">
      <Testables>
         <TestableReference
            skipped = "NO"
            parallelizable = "NO">
{test_ref}         </TestableReference>
      </Testables>
   </TestAction>
   <LaunchAction
      buildConfiguration = "Debug"
      selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB"
      selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB"
      launchStyle = "0"
      useCustomWorkingDirectory = "NO"
      ignoresPersistentStateOnLaunch = "NO"
      debugDocumentVersioning = "YES"
      debugServiceExtension = "internal"
      allowLocationSimulation = "YES">
      <BuildableProductRunnable
         runnableDebuggingMode = "0">
{app_ref}      </BuildableProductRunnable>
   </LaunchAction>
   <ProfileAction
      buildConfiguration = "Release"
      shouldUseLaunchSchemeArgsEnv = "YES"
      savedToolIdentifier = ""
      useCustomWorkingDirectory = "NO"
      debugDocumentVersioning = "YES">
      <BuildableProductRunnable
         runnableDebuggingMode = "0">
{app_ref}      </BuildableProductRunnable>
   </ProfileAction>
   <AnalyzeAction
      buildConfiguration = "Debug">
   </AnalyzeAction>
   <ArchiveAction
      buildConfiguration = "Release"
      revealArchiveInOrganizer = "YES">
   </ArchiveAction>
</Scheme>
'''


WORKSPACE = '''<?xml version="1.0" encoding="UTF-8"?>
<Workspace
   version = "1.0">
   <FileRef
      location = "self:">
   </FileRef>
</Workspace>
'''


def main() -> None:
    (PROJECT / "project.xcworkspace").mkdir(parents=True, exist_ok=True)
    (PROJECT / "xcshareddata" / "xcschemes").mkdir(parents=True, exist_ok=True)
    (PROJECT / "project.pbxproj").write_text(pbxproj())
    (PROJECT / "project.xcworkspace" / "contents.xcworkspacedata").write_text(WORKSPACE)
    (PROJECT / "xcshareddata" / "xcschemes" / f"{APP}.xcscheme").write_text(scheme())
    print(f"wrote {PROJECT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
