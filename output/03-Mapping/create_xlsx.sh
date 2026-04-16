#!/bin/sh

# Order is important => the [Content_Types].xml file must be first
zip -r anonymous_snowballing_process.xlsx \[Content_Types\].xml _rels docProps xl
