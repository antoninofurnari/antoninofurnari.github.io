---
title: "Research"
date: 2022-08-25T17:33:11+02:00
disable_share: true
type: "page"
draft: false
---

I am interested in developing algorithms and wearable systems based on egocentric vision to support users in their daily tasks, be them related to home/personal scenarios or work-related scenarios. I have been working on egocentric (or first-person) vision since the beginning of my PhD (2013) and developed experience on data collection and labeling, the definition of tasks, the development of algorithms, as well as their evaluation.


## Research Highlights
This highlights recent research aligned to my main research interests. Please see the <a href="/publications/" target="_blank">publications page</a> for a full list of publications.


{{< research-list >}}

<script>
    document.addEventListener("DOMContentLoaded", function () {
        // Select all elements with IDs starting with 'bibtex-'
        const preElements = document.querySelectorAll('pre[id^="bibtex-"]');
        
        // Loop through each 'pre' element
        preElements.forEach(preElement => {
            // Extract the identifier (part after 'bibtex-')
            const idSuffix = preElement.id.substring(7);

            // Construct corresponding table id
            const tableId = "bibtexify-" + idSuffix;

            // Ensure the corresponding table exists
            const tableElement = document.getElementById(tableId);
            if (tableElement) {
                // Call bibtexify function with the constructed IDs
                bibtexify("#" + preElement.id, tableId, { hideMissing: true });
            }
        });
    });
</script>